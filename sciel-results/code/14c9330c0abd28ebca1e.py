import time
import sys
from math import gcd

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line (as a normalized tuple) to the set of point indices on it
    from collections import defaultdict
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Also normalize the line by using the "leftmost/bottommost" point as reference
            # A line can be represented by (dx, dy, x0*dx - y0*dy) where (x0,y0) is a point on the line
            # But to make it canonical, we need to handle sign and offset
            # Better: use the line equation ax + by = c where a,b are coprime and c is normalized
            # For direction (dx, dy), the normal is (dy, -dx), so line: dy*x - dx*y = c
            # c = dy*x1 - dx*y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize (a, b, c) so that gcd(a,b,c) = 1 and a > 0 or (a == 0 and b > 0)
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return None
    # Check bounds
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    # Check not all collinear
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Tests several known configurations and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Candidate configurations
    candidates = []
    
    # Config 1: 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 11 on line + 1 off (current baseline)
    pts3 = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts3)
    
    # Config 4: 6 on x-axis, 6 on y-axis (cross)
    pts4 = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)]
    candidates.append(pts4)
    
    # Config 5: Points on two parallel lines (6 each)
    pts5 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
    candidates.append(pts5)
    
    # Config 6: 3 lines of 4 points each (triangle-like)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=x, x=0,1,2,3 -> (0,0),(1,1),(2,2),(3,3) but (0,0) shared
    # Let's do: 3 lines through origin at different slopes, 4 points each, but they share the origin
    # That gives 1 + 3*3 = 10 points. Need 2 more.
    # Alternative: 3 lines not all through one point
    # Line A: y=0, x=0,1,2,3
    # Line B: y=1, x=0,1,2,3  
    # Line C: x=0, y=0,1,2,3 -> but (0,0) and (0,1) already counted
    # This is getting complicated. Let me try a hexagon-based config.
    
    # Config 7: Regular hexagon vertices + center + 3 more
    # Hexagon: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Center: (0,0)
    # 3 more: extend some lines
    # This might not be optimal. Let me try a different approach.
    
    # Config 8: 4 points on each of 3 concurrent lines (through origin)
    # Line 1: y=0: (1,0), (2,0), (3,0), (4,0)
    # Line 2: y=x: (1,1), (2,2), (3,3), (4,4)
    # Line 3: y=2x: (1,2), (2,4), (3,6), (4,8)
    # Plus origin (0,0) and one more point
    pts8 = [[0, 0]]
    for i in range(1, 5):
        pts8.append([i, 0])
        pts8.append([i, i])
        pts8.append([i, 2*i])
    # That's 1 + 12 = 13 points. Need exactly 12.
    # Remove one: pts8 = pts8[:12]
    candidates.append(pts8[:12])
    
    # Config 9: 4 concurrent lines with 3 points each + origin
    # 4 lines through origin, 3 points on each (not counting origin)
    # Total: 1 + 4*3 = 13. Too many.
    # 3 lines through origin, 4 points each (not counting origin): 1 + 3*4 = 13. Still too many.
    # 2 lines through origin, 5 points each: 1 + 2*5 = 11. Need 1 more.
    pts9 = [[0, 0]]
    for i in range(1, 6):
        pts9.append([i, 0])
        pts9.append([i, i])
    # 11 points, add one more
    pts9.append([1, 2])
    candidates.append(pts9)
    
    # Find the best configuration
    best_points = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = pts
    
    # If no valid config found, fallback to the 4x3 grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
