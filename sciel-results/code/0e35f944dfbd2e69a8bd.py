import numpy as np

def construct():
    N = 21
    # Configuration: 6 rows.
    # Rows 0, 2, 4 have 4 circles.
    # Rows 1, 3, 5 have 3 circles.
    # Hexagonal packing.
    
    # Calculate r
    # w = 8r
    # h = 2r + 5 * r * sqrt(3)
    # w + h = 2
    # r * (10 + 5*sqrt(3)) = 2
    
    sqrt3 = np.sqrt(3)
    r = 2.0 / (10.0 + 5.0 * sqrt3)
    
    w = 8.0 * r
    h = 2.0 * r + 5.0 * r * sqrt3
    
    circles = []
    
    # Row 0: 4 circles
    # Centers: r, 3r, 5r, 7r
    # y = r
    for j in range(4):
        x = r + 2.0 * j * r
        y = r
        circles.append((x, y, r))
        
    # Row 1: 3 circles
    # Centers: 2r, 4r, 6r
    # y = r + r*sqrt(3)
    y1 = r + r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append((x, y1, r))
        
    # Row 2: 4 circles
    y2 = r + 2.0 * r * sqrt3
    for j in range(4):
        x = r + 2.0 * j * r
        circles.append((x, y2, r))
        
    # Row 3: 3 circles
    y3 = r + 3.0 * r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append((x, y3, r))
        
    # Row 4: 4 circles
    y4 = r + 4.0 * r * sqrt3
    for j in range(4):
        x = r + 2.0 * j * r
        circles.append((x, y4, r))
        
    # Row 5: 3 circles
    y5 = r + 5.0 * r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append((x, y5, r))
        
    return np.array(circles)
