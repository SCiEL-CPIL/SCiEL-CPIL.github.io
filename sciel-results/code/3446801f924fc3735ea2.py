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
    # Line key: (dx, dy, c) where dx, dy are normalized direction vector (gcd(dx,dy)=1, dx>0 or dx==0,dy>0)
    # and c is the intercept such that dy*x - dx*y = c
    # Wait, the standard line equation is Ax + By + C = 0.
    # For points (x1,y1) and (x2,y2):
    # A = y1 - y2
    # B = x2 - x1
    # C = -(A*x1 + B*y1)
    # Normalize (A, B, C) by GCD and sign.
    
    line_map = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize the coefficients A, B, C
            # Find GCD of A, B, C
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                # This case (A=0, B=0) shouldn't happen for distinct points
                continue
            
            # Enforce a canonical sign
            # We want the first non-zero coefficient to be positive
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
    deadline = 150.0 - 10.0 # Safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            
    # Candidate 1: 3x4 Grid
    # Points (0,0) to (2,3)
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    cnt = count_ordinary_lines(grid_3x4)
    update_best(grid_3x4, cnt)
    
    # Candidate 2: 4x3 Grid (Transposed)
    # Same as above effectively, but let's be explicit
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    cnt = count_ordinary_lines(grid_4x3)
    update_best(grid_4x3, cnt)
    
    # Candidate 3: 3 lines of 4 points, parallel
    # Line 1: y=0, x=0..3
    # Line 2: y=1, x=0..3
    # Line 3: y=2, x=0..3
    # This is the 4x3 grid again.
    
    # Candidate 4: 4 lines of 3 points, forming a "parallelogram" or similar
    # Try a 2x6 grid?
    grid_2x6 = [[x, y] for x in range(2) for y in range(6)]
    cnt = count_ordinary_lines(grid_2x6)
    update_best(grid_2x6, cnt)
    
    # Candidate 5: 6x2 grid
    grid_6x2 = [[x, y] for x in range(6) for y in range(2)]
    cnt = count_ordinary_lines(grid_6x2)
    update_best(grid_6x2, cnt)
    
    # Candidate 6: A specific known configuration with 6 ordinary lines
    # The "Pappus" configuration or similar?
    # Let's try a configuration based on 3 concurrent lines and 3 parallel lines?
    
    # Let's try a "Star of David" like pattern? No, that's 6 points.
    
    # Let's try to perturb the 3x4 grid slightly to see if we can drop to 6 if it's not already 6.
    # First, let's see what the count for 3x4 is.
    # If 3x4 is 5, we are done (impossible, min is 6).
    # If 3x4 is 6, we are done.
    # If 3x4 is > 6, we need to search.
    
    # Let's run a quick local search around the best candidate if time permits
    # and if best_count > 6.
    
    if best_count > 6 and time.time() < deadline - 10:
        import random
        random.seed(42)
        
        # Local search around the best grid
        current_pts = [p[:] for p in best_points]
        current_cnt = best_count
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        old_x, old_y = current_pts[i]
                        new_x, new_y = old_x + dx, old_y + dy
                        current_pts[i] = [new_x, new_y]
                        
                        # Check distinctness
                        if len(set(tuple(p) for p in current_pts)) < 12:
                            current_pts[i] = [old_x, old_y]
                            continue
                        
                        # Check bounds
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            current_pts[i] = [old_x, old_y]
                            continue
                        
                        new_cnt = count_ordinary_lines(current_pts)
                        if new_cnt < current_cnt:
                            current_cnt = new_cnt
                            update_best(current_pts, new_cnt)
                            improved = True
                        else:
                            current_pts[i] = [old_x, old_y]
        
        # Try some random restarts if still > 6
        if best_count > 6 and time.time() < deadline - 5:
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                pts = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
                if len(set(tuple(p) for p in pts)) < 12:
                    continue
                cnt = count_ordinary_lines(pts)
                update_best(pts, cnt)
                if cnt < 6:
                    break
                if cnt == 6:
                    # Local improve to ensure it's stable
                    current_pts = [p[:] for p in pts]
                    current_cnt = cnt
                    for i in range(12):
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                old_x, old_y = current_pts[i]
                                new_x, new_y = old_x + dx, old_y + dy
                                current_pts[i] = [new_x, new_y]
                                if len(set(tuple(p) for p in current_pts)) < 12:
                                    current_pts[i] = [old_x, old_y]
                                    continue
                                new_cnt = count_ordinary_lines(current_pts)
                                if new_cnt < current_cnt:
                                    current_cnt = new_cnt
                                    update_best(current_pts, new_cnt)
                                else:
                                    current_pts[i] = [old_x, old_y]
                                if current_cnt == 6:
                                    break
                            if current_cnt == 6:
                                break
                        if current_cnt == 6:
                            break
                    if best_count == 6:
                        break

    if best_points is None:
        return grid_3x4
    
    return best_points
