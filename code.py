# This script integrates YOLO object detection with object tracking and zone counting.
import cv2  # Importing OpenCV for computer vision tasks.
import pandas as pd  # Importing pandas for data manipulation and analysis.
from ultralytics import YOLO  # Importing YOLO model for object detection.
from tracker import Tracker  # Importing custom object tracker.
import cvzone  # Importing cvzone for drawing utilities.

# Loading the YOLO model for object detection.
model = YOLO('yolov8s.pt')

# Function to handle mouse events for color selection.
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Creating a window and setting up mouse callback.
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Capturing video from the default camera.
cap = cv2.VideoCapture(0)

# Reading the class labels for COCO dataset.
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initializing variables for tracking and counting.
count = 0
persondown = {}
tracker = Tracker()
counter1 = []

personup = {}
counter2 = []
cy1 = 194
cy2 = 220
offset = 6

# Main loop for processing video frames.
while True:
    ret, frame = cap.read()  # Reading a frame from the video.
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))  # Resizing frame for better visualization.

    # Performing object detection using YOLO model.
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)  # Updating object tracker.

    # Processing tracked objects for zone crossing and counting.
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # Drawing centroid of the object.

        # Checking if object crosses a predefined zone.
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            persondown[id] = (cx, cy)

        if id in persondown:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

        if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            personup[id] = (cx, cy)

        if id in personup:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)

    # Drawing the counting zones.
    cv2.line(frame, (cy1, 3), (cy1, 1018), (0, 255, 0), 2)
    cv2.line(frame, (cy2, 5), (cy2, 1019), (0, 255, 255), 2)

    # Displaying the counts on the frame.
    IN = (len(counter1))
    OUT = (len(counter2))
    cvzone.putTextRect(frame, f'IN {IN}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'OUT {OUT}', (50, 160), 2, 2)

    # Showing the processed frame.
    cv2.imshow("RGB", frame)
    
    # Exiting the loop if 'Esc' key is pressed.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Releasing the video capture and destroying all windows.
cap.release()
cv2.destroyAllWindows()
