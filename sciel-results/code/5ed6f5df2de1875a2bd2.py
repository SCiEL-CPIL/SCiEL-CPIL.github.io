import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Map point to index for quick lookup? No, just count lines.
    lines = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0 and dy == 0:
                continue
                
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Normalize direction
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            a = dy
            b = -dx
            c = a * p1[0] + b * p1[1]
            key = (a, b, c)
            
            lines[key].add(i)
            lines[key].add(j)

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Initial configurations
    # 3x4 grid
    config_1 = [[x, y] for x in range(3) for y in range(4)]
    # 4x3 grid
    config_2 = [[x, y] for x in range(4) for y in range(3)]
    # Pappus-like?
    # Just use random restarts from grids
    
    best_points = config_1
    best_score = count_ordinary_lines(best_points)
    
    # Try config 2
    s2 = count_ordinary_lines(config_2)
    if s2 < best_score:
        best_score = s2
        best_points = config_2
        
    if best_score == 6:
        return best_points

    rng = random.Random(42)
    
    # Local Search
    while time.time() < deadline:
        # Strategy: Move one point to a random nearby location
        idx = rng.randint(0, 11)
        old_p = best_points[idx]
        
        # Try to move to a position that creates a collinearity
        # Pick two other points
        i, j = rng.sample([k for k in range(12) if k != idx], 2)
        p1 = best_points[i]
        p2 = best_points[j]
        
        # Line through p1, p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            continue
            
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx_n = dx // g
        dy_n = dy // g
        
        # Try positions on this line
        # p1 + k * (dx_n, dy_n)
        # k should be such that the point is integer and within bounds
        # and not already occupied
        
        # We want to place best_points[idx] on this line.
        # So new_p = p1 + k * (dx_n, dy_n)
        
        # Try small k values
        for k in range(-5, 6):
            if k == 0:
                continue
            nx = p1[0] + k * dx_n
            ny = p1[1] + k * dy_n
            
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
                
            new_p = [nx, ny]
            
            # Check if new_p is already in best_points (other than idx)
            if any(p == new_p for p in best_points if best_points.index(p) != idx):
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            
            new_score = count_ordinary_lines(new_points)
            
            if new_score < best_score:
                best_score = new_score
                best_points = new_points
                if best_score == 6:
                    return best_points
                break # Move to next iteration if improved
                
        # If no improvement found with collinear move, try random jitter
        if time.time() < deadline:
             idx = rng.randint(0, 11)
             old_p = best_points[idx]
             dx = rng.randint(-3, 3)
             dy = rng.randint(-3, 3)
             if dx == 0 and dy == 0:
                 continue
             new_p = [old_p[0] + dx, old_p[1] + dy]
             if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                 continue
             
             new_points = [p[:] for p in best_points]
             new_points[idx] = new_p
             
             # Check duplicates
             if len(set(map(tuple, new_points))) < 12:
                 continue
                 
             new_score = count_ordinary_lines(new_points)
             if new_score < best_score:
                 best_score = new_score
                 best_points = new_points
                 if best_score == 6:
                     return best_points

    return best_points
