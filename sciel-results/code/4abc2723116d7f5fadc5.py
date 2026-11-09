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

def _generate_candidates():
    """Generate a diverse set of candidate configurations."""
    candidates = []
    
    # 1. 11 on line + 1 off (Baseline: 11 ordinary lines)
    candidates.append([[i, 0] for i in range(11)] + [[0, 1]])
    
    # 2. 4x3 Grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3. 3x4 Grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 4. 2x6 Grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 5. 6x2 Grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 6. Pencils (Lines through origin)
    # 3 lines, 4 points each (including origin) -> 1 + 3*3 = 10? No.
    # Let's do 4 lines, 3 points each (including origin) -> 1 + 4*2 = 9? No.
    # We need 12 points.
    # Try 3 lines through origin. 4 points on each line.
    # Points: (0,0), (1,0), (2,0), (3,0), (1,1), (2,2), (3,3), (1,2), (2,4), (3,6), (1,3), (2,6)
    pts_pencil = [[0,0]]
    slopes = [(0,1), (1,1), (2,1)]
    for dy, dx in slopes:
        for t in range(1, 4):
            pts_pencil.append([t*dx, t*dy])
    # That's 1 + 3*3 = 10 points. Add 2 more?
    # Let's try 4 lines, 3 points each.
    pts_pencil2 = [[0,0]]
    slopes2 = [(0,1), (1,1), (2,1), (3,1)]
    for dy, dx in slopes2:
        for t in range(1, 3):
            pts_pencil2.append([t*dx, t*dy])
    # 1 + 4*2 = 9 points.
    # Add 3 more?
    # Let's just add random points or extend lines.
    candidates.append(pts_pencil)
    
    # 7. Points on Parabola y = x^2
    candidates.append([[x, x*x] for x in range(12)])
    
    # 8. Points on Ellipse approx (Integer points)
    # x^2 + y^2 = r^2 is hard.
    # Let's try y = x^2 + 1
    candidates.append([[x, x*x + 1] for x in range(12)])
    
    # 9. "Almost" 11+1: Move the off-point to create collinearity
    # 10 on x-axis, 2 off
    candidates.append([[i, 0] for i in range(10)] + [[0, 1], [1, 1]])
    candidates.append([[i, 0] for i in range(10)] + [[0, 1], [2, 1]])
    candidates.append([[i, 0] for i in range(10)] + [[0, 1], [0, 2]])
    
    # 10. 9 on line, 3 off (forming a line themselves)
    candidates.append([[i, 0] for i in range(9)] + [[0, 1], [1, 1], [2, 1]])
    
    # 11. 8 on line, 4 off (forming a line themselves)
    candidates.append([[i, 0] for i in range(8)] + [[0, 1], [1, 1], [2, 1], [3, 1]])
    
    # 12. 6 on x-axis, 6 on y-axis (Cross)
    candidates.append([[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)])
    
    # 13. 6 on x-axis, 6 on line y=1 (Parallel)
    candidates.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(6)])
    
    # 14. 6 on x-axis, 6 on line y=1, shifted
    candidates.append([[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)])
    
    return candidates

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    candidates = _generate_candidates()
    
    best_points = None
    best_count = float('inf')
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    if best_points is None:
        # Fallback
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = 11
    
    # Local Search
    # Strategy: Move one point at a time to a random nearby position or a position
    # that creates a new 3-point line.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(12):
            if time.time() > deadline - 5:
                break
            orig_x, orig_y = best_points[i]
            
            # Try small perturbations
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [orig_x + dx, orig_y + dy]
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
                        improved = True
                        break # Restart inner loops? Or just break dy loop
                if improved:
                    break
            if improved:
                break # Restart outer loop
    
    # Second Phase: Larger random jumps
    if best_points is not None and time.time() < deadline - 5:
        rng = random.Random(42)
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            i = rng.randint(0, 11)
            orig_x, orig_y = best_points[i]
            # Try a larger random move
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                new_x = orig_x + rng.randint(-50, 50)
                new_y = orig_y + rng.randint(-50, 50)
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
