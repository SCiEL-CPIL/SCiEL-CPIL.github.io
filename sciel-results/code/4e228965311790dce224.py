import itertools
import time
import random
from math import gcd
from collections import Counter

def normalize_direction(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return (0, 0)
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical form: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)

def get_line_key(p1, p2):
    """Get a canonical key for the line through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line key: (direction, offset) where offset is computed from a reference point
    # Use the formula: for line ax + by = c, normalize (a,b)
    a = dy
    b = -dx
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
    # Ensure canonical sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * p1[0] + b * p1[1]
    return (a, b, c)

def count_4point_lines(points):
    """Count lines with exactly 4 points, and check no 5+ collinear."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    has_5 = False
    for key, pts in line_points.items():
        if len(pts) >= 5:
            has_5 = True
        if len(pts) == 4:
            count_4 += 1
    
    return count_4, has_5

def score_config(points):
    """Return score: number of 4-point lines if valid, -1 if invalid."""
    count_4, has_5 = count_4point_lines(points)
    if has_5:
        return -1
    return count_4

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_score = score_config(points)
    best_points = [p[:] for p in points]
    
    if best_score < 0:
        # Should not happen for 4x4 grid
        return best_points
    
    # Local search: try moving each point to nearby positions
    random.seed(42)
    
    coord_range = 1000
    step_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    
    while time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            orig = points[i][:]
            # Try various moves
            for step in step_sizes:
                for dx in range(-step, step + 1):
                    for dy in range(-step, step + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig[0] + dx
                        new_y = orig[1] + dy
                        if abs(new_x) > coord_range or abs(new_y) > coord_range:
                            continue
                        points[i] = [new_x, new_y]
                        # Check distinctness
                        valid = True
                        for j in range(16):
                            if j != i and points[j] == points[i]:
                                valid = False
                                break
                        if not valid:
                            points[i] = orig
                            continue
                        s = score_config(points)
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in points]
                            improved = True
                        else:
                            points[i] = orig
                if time.time() > deadline:
                    break
            points[i] = orig
        
        if not improved:
            # Try random restarts or larger moves
            # Occasionally do a random perturbation
            if random.random() < 0.3:
                # Perturb a random point more aggressively
                i = random.randint(0, 15)
                orig = points[i][:]
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    new_x = random.randint(-coord_range, coord_range)
                    new_y = random.randint(-coord_range, coord_range)
                    points[i] = [new_x, new_y]
                    valid = True
                    for j in range(16):
                        if j != i and points[j] == points[i]:
                            valid = False
                            break
                    if not valid:
                        continue
                    s = score_config(points)
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in points]
                        improved = True
                    else:
                        points[i] = orig
                points[i] = orig
            
            if not improved and best_score >= 10:
                # We've likely converged; do a few more random restarts
                break
    
    return best_points
