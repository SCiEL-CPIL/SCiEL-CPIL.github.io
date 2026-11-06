import time
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def _generate_hesse_config():
    """
    The Hesse configuration (9_4) consists of 9 points and 12 lines.
    It can be represented as the 3x3 grid.
    To get 12 points, we can add 3 more points carefully.
    However, the 3x4 grid is often cited as having 6 ordinary lines.
    Let's generate a few structured candidates.
    """
    candidates = []
    
    # 1. 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_3x4)
    
    # 2. 4x3 Grid (same as above, just checking)
    grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_4x3)
    
    # 3. Affine Plane AG(2, 3) plus 3 points?
    # AG(2,3) has 9 points. Lines have 3 points.
    # If we add 3 points, we might create ordinary lines.
    # Let's try adding points at infinity or specific locations.
    
    # 4. Hesse configuration (9 points) + 3 points at infinity?
    # The 9 points are (x,y) in {0,1,2}^2.
    # The 12 lines are the 4 lines in each direction (slope 0, 1, 2, inf) passing through the grid.
    # Wait, in AG(2,3), there are 12 lines total. Each line has 3 points.
    # Total incidences: 12 * 3 = 36. Points: 9. Each point is on 4 lines.
    # If we take these 9 points, there are 0 ordinary lines among them (all lines have 3 points).
    # But we need 12 points.
    
    # Let's try a configuration based on the Pappus configuration or similar.
    
    # 5. Two parallel lines with 4 points each, and 4 points on a transversal?
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: x=0, y=0,1,2,3 -> (0,0), (0,1) are shared.
    # This is getting complicated. Let's stick to grids and small perturbations.
    
    # 6. 2x6 Grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_2x6)
    
    # 7. 1x12 is invalid (all collinear).
    
    # 8. Triangle with 4 points on each side (no shared vertices)
    # Side 1: (0,0), (1,0), (2,0), (3,0)
    # Side 2: (0,1), (0,2), (0,3), (0,4)
    # Side 3: (1,4), (2,4), (3,4), (4,4)
    # Check distinctness:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (0,2), (0,3), (0,4)
    # (1,4), (2,4), (3,4), (4,4)
    # All distinct? Yes.
    # Are they all collinear? No.
    triangle = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [0,2], [0,3], [0,4],
        [1,4], [2,4], [3,4], [4,4]
    ]
    candidates.append(triangle)
    
    # 9. A configuration from the "few ordinary lines" literature.
    # For n=12, the minimum is 6.
    # The 3x4 grid is known to have 6 ordinary lines.
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    candidates = _generate_hesse_config()
    
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        if len(set(map(tuple, cand))) != 12:
            continue
        if len(set([_line_key(cand[0], cand[i]) for i in range(1, 12)])) == 1:
            continue
            
        s = _count_ordinary(cand)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
            
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback to 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_points)
        
    return best_points
