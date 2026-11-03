import numpy as np

N = 21

def construct():
    # Hexagonal close-packed arrangement for 21 circles
    # Rows: 4, 3, 4, 3, 4, 3 (total 21)
    # This minimizes the bounding box perimeter compared to square grid
    r = 0.10715  # radius computed from perimeter constraint 2(w+h) <= 4
    
    # Row 0: 4 circles at y = r
    # Row 1: 3 circles at y = r + sqrt(3)*r (shifted right by r)
    # Row 2: 4 circles at y = r + 2*sqrt(3)*r
    # Row 3: 3 circles at y = r + 3*sqrt(3)*r
    # Row 4: 4 circles at y = r + 4*sqrt(3)*r
    # Row 5: 3 circles at y = r + 5*sqrt(3)*r
    
    sqrt3 = np.sqrt(3)
    circles = []
    
    # Row 0: 4 circles
    for i in range(4):
        x = r + 2*r*i
        y = r
        circles.append([x, y, r])
    
    # Row 1: 3 circles (shifted by r)
    for i in range(3):
        x = 2*r + 2*r*i
        y = r + sqrt3*r
        circles.append([x, y, r])
    
    # Row 2: 4 circles
    for i in range(4):
        x = r + 2*r*i
        y = r + 2*sqrt3*r
        circles.append([x, y, r])
    
    # Row 3: 3 circles (shifted by r)
    for i in range(3):
        x = 2*r + 2*r*i
        y = r + 3*sqrt3*r
        circles.append([x, y, r])
    
    # Row 4: 4 circles
    for i in range(4):
        x = r + 2*r*i
        y = r + 4*sqrt3*r
        circles.append([x, y, r])
    
    # Row 5: 3 circles (shifted by r)
    for i in range(3):
        x = 2*r + 2*r*i
        y = r + 5*sqrt3*r
        circles.append([x, y, r])
    
    return np.array(circles[:N])
