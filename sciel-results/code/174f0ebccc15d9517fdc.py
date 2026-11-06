import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0
    
    n = 32
    points = []
    point_set = set()
    
    # Generate spiral points
    # Ring 0
    points.append((0, 0))
    point_set.add((0, 0))
    
    if len(points) >= n:
        return [list(p) for p in points]
        
    # Directions for the spiral
    # 0: (1, 0)
    # 1: (0, 1)
    # 2: (-1, 1)
    # 3: (-1, 0)
    # 4: (0, -1)
    # 5: (1, -1)
    
    # We construct rings manually to ensure the correct spiral order
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for p in ring1:
        if len(points) >= n:
            break
        if p not in point_set:
            points.append(p)
            point_set.add(p)
            
    if len(points) >= n:
        return [list(p) for p in points]

    # Ring 2
    # Starts at (2, -1) which is neighbor of (1, -1)
    # Order: (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2)
    # Wait, let's verify the "next" point from (1, -1) is (2, -1).
    # And the sequence continues CCW.
    ring2 = [
        (2, -1), (2, 0), (1, 1), (0, 2), 
        (-1, 2), (-2, 1), (-2, 0), (-1, -1), 
        (0, -2), (1, -2), (2, -2)
    ]
    # Check if (2,-2) is the last.
    # Neighbors of (2,-2): (1,-2) [in ring2], (2,-1) [in ring2], (3,-2) [ring3], (3,-3) [ring3], (2,-3) [ring3], (1,-3) [ring3].
    # So (2,-2) is a corner.
    for p in ring2:
        if len(points) >= n:
            break
        if p not in point_set:
            points.append(p)
            point_set.add(p)
            
    if len(points) >= n:
        return [list(p) for p in points]

    # Ring 3
    # Starts at (3, -2) which is neighbor of (2, -2)
    # Sequence:
    # (3, -2), (3, -3), (2, -3), (1, -3), (0, -3), 
    # (-1, -2), (-2, -1), (-3, 0), (-3, 1), (-3, 2), 
    # (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 2), 
    # (2, 1), (3, 0), (3, -1)
    
    ring3 = [
        (3, -2), (3, -3), (2, -3), (1, -3), (0, -3),
        (-1, -2), (-2, -1), (-3, 0), (-3, 1), (-3, 2),
        (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 2),
        (2, 1), (3, 0), (3, -1)
    ]
        
    for p in ring3:
        if len(points) >= n:
            break
        if p not in point_set:
            points.append(p)
            point_set.add(p)
            
    # If we still need more (unlikely for n=32), we can generate Ring 4
    # But 1 + 6 + 12 + 18 = 37. We need 32.
    # So we will stop in Ring 3.
    
    return [list(p) for p in points[:n]]
