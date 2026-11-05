import time
import math
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines that contain exactly two of the given points.
    Points are a list of [x, y] integers.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to set of point indices
    line_map = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize the coefficients A, B, C
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                continue
            
            # Enforce a canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in line_map:
                line_map[key] = set()
            
            line_map[key].add(i)
            line_map[key].add(j)
    
    ordinary_count = 0
    for key, indices in line_map.items():
        if len(indices) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            # If we hit the theoretical minimum, we can stop early
            if count == 6:
                return True
        return False

    # Candidate 1: 4x3 Grid
    # This configuration is known to be highly structured.
    # Points: (x, y) for x in 0..3, y in 0..2
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    cnt = count_ordinary_lines(grid_4x3)
    if update_best(grid_4x3, cnt):
        return best_points

    # Candidate 2: 3x4 Grid (Transposed)
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    cnt = count_ordinary_lines(grid_3x4)
    if update_best(grid_3x4, cnt):
        return best_points

    # Candidate 3: 2x6 Grid
    grid_2x6 = [[x, y] for x in range(2) for y in range(6)]
    cnt = count_ordinary_lines(grid_2x6)
    if update_best(grid_2x6, cnt):
        return best_points

    # Candidate 4: 6x2 Grid
    grid_6x2 = [[x, y] for x in range(6) for y in range(2)]
    cnt = count_ordinary_lines(grid_6x2)
    if update_best(grid_6x2, cnt):
        return best_points

    # If none of the simple grids achieved 6, we perform a structured local search.
    # We focus on the best grid found so far and try to "snap" points to nearby integer
    # coordinates that might form additional collinearities.
    
    if best_points is None:
        return grid_4x3

    current_pts = [p[:] for p in best_points]
    current_cnt = best_count
    
    # Deterministic Local Search:
    # Try moving each point to all nearby integer positions within radius 2.
    # This is much faster and more reliable than random restarts for small grids.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(12):
            ox, oy = current_pts[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    
                    # Bounds check
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                        
                    # Tentative move
                    old_pt = current_pts[i]
                    current_pts[i] = [nx, ny]
                    
                    # Distinctness check
                    if len(set(tuple(p) for p in current_pts)) < 12:
                        current_pts[i] = old_pt
                        continue
                        
                    new_cnt = count_ordinary_lines(current_pts)
                    
                    if new_cnt < current_cnt:
                        current_cnt = new_cnt
                        update_best(current_pts, new_cnt)
                        improved = True
                        if current_cnt == 6:
                            return best_points
                    else:
                        current_pts[i] = old_pt
                        
    return best_points
