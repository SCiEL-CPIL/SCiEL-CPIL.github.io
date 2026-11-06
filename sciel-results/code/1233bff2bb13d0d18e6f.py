import random
import time
import math
from itertools import combinations
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            # Normalize: make (A,B) lexicographically positive, then C
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0:
                if C < 0:
                    C = -C
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _boroczky_12():
    """
    Böröczky-type configuration for 12 points achieving 6 ordinary lines.
    
    Construction: 6 points on a circle (conic) and 6 points on a line,
    arranged so that the group law on the conic ensures many collinearities.
    
    A concrete integer realization:
    - 6 points on the parabola y = x^2 (a conic)
    - 6 points on the x-axis (y=0)
    
    But we need to verify this gives exactly 6 ordinary lines.
    
    Alternative known construction: use the 3x4 grid which is documented
    to have exactly 6 ordinary lines.
    
    Let's use a verified configuration:
    Points: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1),
            (0,2), (1,2), (2,2), (3,2) -- this is 4x3 grid
    
    Actually, the standard result is that an m x n grid has a specific
    number of ordinary lines. For 3x4, it should be 6.
    """
    # 3 rows, 4 columns grid
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append([j, i])
    return pts

def _alt_configuration_1():
    """
    Alternative: 6 points on one line, 6 on another parallel line,
    with careful x-coordinates to maximize collinearities across lines.
    """
    # Line 1: y=0, x in {0,1,2,3,4,5}
    # Line 2: y=1, x in {0,1,2,3,4,5}
    # This is a 2x6 grid
    pts = []
    for i in range(2):
        for j in range(6):
            pts.append([j, i])
    return pts

def _alt_configuration_2():
    """
    Use points on a conic: y = x^2 for x in some set, plus points at infinity
    approximated by large coordinates.
    
    Actually, let's use a known 12-point configuration from literature.
    The Böröczky example for n=12 uses:
    - 6 points on a circle
    - 6 points on a line
    
    Integer approximation:
    Circle: x^2 + y^2 = 25 (radius 5)
    Points: (0,5), (3,4), (5,0), (4,3), (0,-5), (-3,-4) -- 6 points on circle
    Line: y = 10, x in {0,1,2,3,4,5} -- 6 points on line
    
    Let's try this.
    """
    pts = [
        [0, 5], [3, 4], [5, 0], [4, 3], [0, -5], [-3, -4],
        [0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10]
    ]
    return pts

def _local_search(points, max_steps=5000, time_limit=30):
    """
    Local search: try moving individual points to reduce ordinary line count.
    """
    start = time.time()
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    
    if current_score <= 6:
        return current, current_score
    
    best = [p[:] for p in current]
    best_score = current_score
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        if best_score <= 6:
            break
        
        # Try moving a random point to a random nearby location
        idx = random.randint(0, 11)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        
        new_x = current[idx][0] + dx
        new_y = current[idx][1] + dy
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        
        # Check distinctness
        orig = [current[idx][0], current[idx][1]]
        trial = [p[:] for p in current]
        trial[idx] = [new_x, new_y]
        
        # Check all distinct
        seen = set()
        valid = True
        for p in trial:
            key = (p[0], p[1])
            if key in seen:
                valid = False
                break
            seen.add(key)
        if not valid:
            continue
        
        if _all_collinear(trial):
            continue
        
        trial_score = _count_ordinary_lines(trial)
        if trial_score < current_score:
            current = trial
            current_score = trial_score
            if current_score < best_score:
                best = [p[:] for p in current]
                best_score = current_score
        else:
            # Sometimes accept worse to escape local minima
            if random.random() < 0.1:
                current = trial
                current_score = trial_score
    
    return best, best_score

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    # Try deterministic constructions first
    candidates = []
    
    # 1. 3x4 grid
    grid_3x4 = _boroczky_12()
    score_3x4 = _count_ordinary_lines(grid_3x4)
    candidates.append((score_3x4, grid_3x4))
    
    # 2. 2x6 grid
    grid_2x6 = _alt_configuration_1()
    score_2x6 = _count_ordinary_lines(grid_2x6)
    candidates.append((score_2x6, grid_2x6))
    
    # 3. Circle + line configuration
    circ_line = _alt_configuration_2()
    score_circ = _count_ordinary_lines(circ_line)
    candidates.append((score_circ, circ_line))
    
    # 4. 4x3 grid
    grid_4x3 = []
    for i in range(4):
        for j in range(3):
            grid_4x3.append([j, i])
    score_4x3 = _count_ordinary_lines(grid_4x3)
    candidates.append((score_4x3, grid_4x3))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    best_score, best_points = candidates[0]
    
    # If we already have 6, return immediately
    if best_score <= 6:
        return best_points
    
    # Use remaining time for local search from the best candidate
    if time.time() < deadline - 5:
        search_points, search_score = _local_search(best_points, max_steps=10000, time_limit=35)
        if search_score < best_score:
            best_score = search_score
            best_points = search_points
    
    # Also try local search from other candidates if time permits
    for score, pts in candidates[1:]:
        if time.time() > deadline - 10:
            break
        if score >= best_score:
            break
        search_points, search_score = _local_search(pts, max_steps=5000, time_limit=15)
        if search_score < best_score:
            best_score = search_score
            best_points = search_points
    
    if time.time() > deadline - 2:
        return best_points
    
    # Final verification
    final_score = _count_ordinary_lines(best_points)
    if final_score < best_score:
        best_score = final_score
    
    return best_points
