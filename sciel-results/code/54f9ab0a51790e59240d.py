"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a canonical key for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        # vertical line
        return (0, 1, x1)
    if dy == 0:
        # horizontal line
        return (1, 0, y1)
    # normalize: make dx positive, then reduce by gcd
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dy, -dx, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line to the set of points on it
    line_to_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _random_points(n=12, coord_range=50):
    """Generate n random integer points."""
    points = []
    seen = set()
    while len(points) < n:
        x = random.randint(-coord_range, coord_range)
        y = random.randint(-coord_range, coord_range)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    return points

def _is_valid(points):
    """Check if points are valid: 12 distinct, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        p0, p1, p2 = points[0], points[1], points[2]
        (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if cross == 0:
            # Check if all are collinear
            all_col = True
            for p in points[3:]:
                x3, y3 = p
                cross3 = (x1 - x0) * (y3 - y0) - (y1 - y0) * (x3 - x0)
                if cross3 != 0:
                    all_col = False
                    break
            if all_col:
                return False
    return True

def _local_search(points, max_iters=5000, coord_range=100):
    """Local search to minimize ordinary lines."""
    best_points = [list(p) for p in points]
    best_score = _count_ordinary(best_points)
    current_points = [list(p) for p in points]
    current_score = best_score
    
    for it in range(max_iters):
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_p = list(current_points[idx])
        # Move it to a new random position
        new_x = random.randint(-coord_range, coord_range)
        new_y = random.randint(-coord_range, coord_range)
        current_points[idx] = [new_x, new_y]
        
        # Check validity
        if not _is_valid(current_points):
            current_points[idx] = old_p
            continue
        
        new_score = _count_ordinary(current_points)
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            # Accept with some probability (simple hill climbing with occasional acceptance)
            if random.random() < 0.1:
                current_score = new_score
            else:
                current_points[idx] = old_p
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Try a few initial configurations
    initial_configs = []
    
    # 3x4 grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    initial_configs.append(grid)
    
    # Random starts
    random.seed(42)
    for _ in range(5):
        initial_configs.append(_random_points(12, 50))
    
    # Also try some structured configurations
    # Points on a conic-like structure
    # Try 4 lines of 3 points
    # Configuration: (0,0),(1,0),(2,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (3,0),(3,1),(3,2)
    # That's just a 4x3 grid again
    
    # Try a different arrangement: 3 rows of 4 with some shifts
    # Or use the "Hesse" type configuration
    
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
        
        # Local search from this config
        if time.time() > deadline - 5.0:
            break
        ls_points, ls_score = _local_search(config, max_iters=3000, coord_range=100)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 10.0:
        config = _random_points(12, 200)
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
        
        ls_points, ls_score = _local_search(config, max_iters=2000, coord_range=200)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    if best_points is None:
        # Fallback: return the grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
