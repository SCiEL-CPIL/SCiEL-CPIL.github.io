import time
import sys
from math import gcd
from collections import defaultdict
import random

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line (as a normalized tuple) to the set of point indices on it
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
    Tests several known configurations and performs local search to improve.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Candidate configurations
    candidates = []
    
    # Config 1: 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 11 on line + 1 off
    pts3 = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts3)
    
    # Config 4: 6 on x-axis, 6 on y-axis (cross)
    pts4 = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)]
    candidates.append(pts4)
    
    # Config 5: Points on two parallel lines (6 each)
    pts5 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
    candidates.append(pts5)
    
    # Config 8: 4 points on each of 3 concurrent lines (through origin)
    pts8 = [[0, 0]]
    for i in range(1, 5):
        pts8.append([i, 0])
        pts8.append([i, i])
        pts8.append([i, 2*i])
    candidates.append(pts8[:12])
    
    # Config 9: 2 lines through origin, 5 points each + 1 more
    pts9 = [[0, 0]]
    for i in range(1, 6):
        pts9.append([i, 0])
        pts9.append([i, i])
    pts9.append([1, 2])
    candidates.append(pts9)
    
    # Config 10: 10 on line + 2 off
    for off1_x, off1_y, off2_x, off2_y in [(0,1,1,1), (0,1,2,1), (0,1,0,2), (1,1,2,1), (0,1,1,2)]:
        pts10 = [[i, 0] for i in range(10)] + [[off1_x, off1_y], [off2_x, off2_y]]
        candidates.append(pts10)
    
    # Config 11: 9 on line + 3 off
    pts11 = [[i, 0] for i in range(9)] + [[0, 1], [1, 1], [2, 1]]
    candidates.append(pts11)
    
    # Config 12: 8 on line + 4 off
    pts12 = [[i, 0] for i in range(8)] + [[0, 1], [1, 1], [2, 1], [3, 1]]
    candidates.append(pts12)
    
    # Config 13: Two parallel lines with 6 each, shifted
    pts13 = [[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)]
    candidates.append(pts13)
    
    # New: 3x4 grid variant with different spacing
    for sx, sy in [(1,1), (1,2), (2,1), (2,2), (3,1), (1,3)]:
        pts_grid = [[x*sx, y*sy] for x in range(4) for y in range(3)]
        candidates.append(pts_grid)
    
    # New: Points on a parabola y=x^2
    pts_para = [[x, x*x] for x in range(12)]
    candidates.append(pts_para)
    
    # New: Points on a line plus points forming another structure
    # 6 on y=0, 6 on y=1 but with different x to create diagonals
    for shift in range(0, 4):
        pts_par = [[i, 0] for i in range(6)] + [[i+shift, 1] for i in range(6)]
        candidates.append(pts_par)
    
    # New: 4 lines through origin with 3 points each (sharing origin)
    # Lines: y=0, y=x, y=2x, y=3x
    pts_conc = [[0, 0]]
    for i in range(1, 4):
        pts_conc.append([i, 0])
        pts_conc.append([i, i])
        pts_conc.append([i, 2*i])
    # That's 10 points, add 2 more on y=3x
    pts_conc.append([1, 3])
    pts_conc.append([2, 6])
    candidates.append(pts_conc)
    
    # New: 5 lines through origin with 2-3 points each
    pts_5line = [[0, 0]]
    slopes = [(0,1), (1,1), (2,1), (1,2), (3,1)]  # (dy, dx)
    for dy, dx in slopes:
        for t in range(1, 3):
            pts_5line.append([t*dx, t*dy])
    # That's 1 + 5*2 = 11, add one more
    pts_5line.append([3, 3])
    candidates.append(pts_5line[:12])
    
    # Find the best configuration from candidates
    best_points = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    # If no valid config found, fallback
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = 11
    
    # Local search: try to improve by moving one point at a time
    if best_points is not None and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                orig_x, orig_y = best_points[i]
                # Try larger range of perturbations
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        # Check distinctness
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
    
    # Second phase: try larger jumps
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Pick a random point and try a larger move
            i = random.randint(0, 11)
            orig_x, orig_y = best_points[i]
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                new_x = orig_x + random.randint(-10, 10)
                new_y = orig_y + random.randint(-10, 10)
                new_pt = [new_x, new_y]
                if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                    continue
                if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                    continue
                trial = [list(p) for p in best_points]
                trial[i] = new_pt
                count = _try_config(trial)
                if count is not None and count < best_count:
                    best_points = trial
                    best_count = count
    
    return best_points
