import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Strategy: Two parallel lines at y=0 and y=1 with consecutive integer x-coordinates.
    # 15 points on each line, x from 0 to 14.
    # For doubled area d (1..14): count = 2 * 15 * (15 - d)
    # For d=1: 2 * 15 * 14 = 420 triangles with doubled area = 1.
    
    k = 15
    m = 15
    
    points = []
    for i in range(k):
        points.append((i, 0))
    for j in range(m):
        points.append((j, 1))
        
    # Verify distinctness and bounds
    if len(set(points)) != 30:
        return [(x, y) for x in range(6) for y in range(5)]
        
    for p in points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]
    
    # Check deadline before returning
    if time.time() > deadline:
        return [(x, y) for x in range(6) for y in range(5)]
    
    return points
