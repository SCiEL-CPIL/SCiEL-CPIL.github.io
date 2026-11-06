"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize direction vector
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Reduce by gcd
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Line key: (direction, offset) where offset is the line's position
    # For line with direction (dx,dy) through (x1,y1), the offset is (dy*x1 - dx*y1) / (dx^2 + dy^2)
    # But to keep integers, use: offset = dy*x1 - dx*y1
    offset = dy * x1 - dx * y1
    return (dx, dy, offset)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map each line to the set of points on it
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _eval_config(points):
    """Evaluate a configuration: return (ordinary_count, is_valid)."""
    n = len(points)
    if n != 12:
        return (float('inf'), False)
    
    # Check distinctness
    if len(set(map(tuple, points))) != 12:
        return (float('inf'), False)
    
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (float('inf'), False)
    
    # Check not all collinear
    if n >= 3:
        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            # Check if all points are on this line
            all_collinear = True
            for i in range(3, n):
                xi, yi = points[i]
                c = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
                if c != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return (float('inf'), False)
    
    ordinary = _count_ordinary(points)
    return (ordinary, True)

def _random_point_in_range(lo, hi):
    """Generate a random integer point in [lo, hi] x [lo, hi]."""
    import random
    x = random.randint(lo, hi)
    y = random.randint(lo, hi)
    return [x, y]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with the 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score, _ = _eval_config(best_points)
    
    # Try some known good configurations
    candidates = []
    
    # 3x4 grid (current)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Try a 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # Try a 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # Try some structured configurations
    # Points on a conic or other algebraic curve might have fewer ordinary lines
    # For example, 12 points on a parabola y = x^2
    candidates.append([[x, x*x] for x in range(12)])
    
    # 12 points on a circle (approximate)
    import math
    circle_pts = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        x = round(100 * math.cos(angle))
        y = round(100 * math.sin(angle))
        circle_pts.append([x, y])
    # Check distinctness
    if len(set(map(tuple, circle_pts))) == 12:
        candidates.append(circle_pts)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score, valid = _eval_config(cand)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    # Local search: try to improve by moving individual points
    import random
    random.seed(42)
    
    # Search in a bounded region for efficiency
    search_lo = -50
    search_hi = 50
    
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Create a copy of the current best
        current = [p[:] for p in best_points]
        
        # Try to improve by moving one random point
        idx = random.randint(0, 11)
        old_point = current[idx][:]
        new_x = random.randint(search_lo, search_hi)
        new_y = random.randint(search_lo, search_hi)
        current[idx] = [new_x, new_y]
        
        # Check if it's a valid improvement
        score, valid = _eval_config(current)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in current]
        else:
            # Revert
            current[idx] = old_point
    
    # Final polish: try more targeted moves
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Try moving a point that's on an ordinary line
        # First, identify ordinary lines
        line_points = defaultdict(set)
        n = len(best_points)
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(best_points[i], best_points[j])
                if key is not None:
                    line_points[key].add(i)
                    line_points[key].add(j)
        
        ordinary_lines = [key for key, pts in line_points.items() if len(pts) == 2]
        
        if not ordinary_lines:
            break
        
        # Pick a random ordinary line and try to move one of its points
        line = random.choice(ordinary_lines)
        pts_on_line = line_points[line]
        idx = random.choice(list(pts_on_line))
        
        current = [p[:] for p in best_points]
        old_point = current[idx][:]
        
        # Try moving the point to break this ordinary line
        # Move it to a position that might create more collinearities
        for _ in range(10):
            new_x = random.randint(search_lo, search_hi)
            new_y = random.randint(search_lo, search_hi)
            current[idx] = [new_x, new_y]
            
            score, valid = _eval_config(current)
            if valid and score < best_score:
                best_score = score
                best_points = [p[:] for p in current]
                break
            current[idx] = old_point
    
    return best_points
# EVOLVE-BLOCK-END
