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

def _generate_candidates():
    """
    Generate a diverse set of candidate configurations for 12 points.
    Focus on configurations with high symmetry and known low ordinary line counts.
    """
    candidates = []
    
    # 1. 3x4 Grid (Known to have 6 ordinary lines)
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_3x4)
    
    # 2. 4x3 Grid (Same as above, transposed)
    grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_4x3)
    
    # 3. Pappus Configuration (9 points) + 3 points
    # Pappus theorem: 9 points, 12 lines. 
    # Standard Pappus points can be generated from two lines with 3 points each.
    # Let's try a specific integer realization of Pappus plus 3 extra points.
    # Pappus points from lines y=0 (x=0,1,2) and y=1 (x=0,1,2)
    # Intersections of cross-lines.
    # This is complex to generate generically, so we'll use a known small config.
    
    # 4. Hesse Configuration (9 points) + 3 points
    # Hesse config is 9 points, 12 lines, 3 points per line.
    # Points: (x,y) in {0,1,2}^2
    # If we add 3 points at infinity (conceptually), we need finite points.
    # Let's try adding points to the 3x3 grid.
    hesse_base = [[x, y] for x in range(3) for y in range(3)]
    
    # Try adding points (3,0), (0,3), (3,3)
    cand_hesse1 = hesse_base + [[3,0], [0,3], [3,3]]
    candidates.append(cand_hesse1)
    
    # Try adding points (3,1), (1,3), (2,2) - wait (2,2) is in base.
    # Try (3,1), (1,3), (2,4)
    cand_hesse2 = hesse_base + [[3,1], [1,3], [2,4]]
    candidates.append(cand_hesse2)
    
    # 5. Two parallel lines with 4 points each, and 4 points on a transversal?
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: x=2, y=0,1,2,3 -> (2,0), (2,1) are shared.
    # This creates a grid-like structure.
    cand_transversal = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [1,1], [2,1], [3,1],
        [2,2], [2,3], [4,2], [4,3]
    ]
    candidates.append(cand_transversal)
    
    # 6. A configuration from the "few ordinary lines" literature for n=12.
    # The 3x4 grid is the standard example.
    
    # 7. Try a "broken" grid to see if we can reduce ordinary lines.
    # Start with 3x4 grid and move one point.
    grid_3x4_mod = [[x, y] for x in range(4) for y in range(3)]
    # Move (3,2) to (3,3)
    grid_3x4_mod[11] = [3,3]
    candidates.append(grid_3x4_mod)
    
    # Move (3,2) to (4,2)
    grid_3x4_mod2 = [[x, y] for x in range(4) for y in range(3)]
    grid_3x4_mod2[11] = [4,2]
    candidates.append(grid_3x4_mod2)
    
    # 8. Symmetric configuration around origin
    # 12 points: 4 on each of 3 lines through origin? No, that's 12 points on 3 lines.
    # If 4 points on each of 3 concurrent lines, ordinary lines are the lines connecting points on different lines.
    # This is likely high.
    
    # 9. 2x6 Grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_2x6)
    
    # 10. Triangle with 4 points on each side (no shared vertices)
    triangle = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [0,2], [0,3], [0,4],
        [1,4], [2,4], [3,4], [4,4]
    ]
    candidates.append(triangle)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    candidates = _generate_candidates()
    
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        if len(set(map(tuple, cand))) != 12:
            continue
        # Check not all collinear
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
