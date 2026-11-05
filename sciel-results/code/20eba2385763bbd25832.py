import time
import random
from math import gcd

def count_lines(points):
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
                
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dx_n * x1 - dy_n * y1
            key = (dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    valid = True
    score = 0
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            score += 1
            
    return valid, score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    valid, best_score = count_lines(best_points)
    if not valid:
        best_score = -1

    # List of initial configurations to try
    inits = [
        [[x, y] for x in range(4) for y in range(4)],
        [[x, y] for x in range(-2, 2) for y in range(-2, 2)],
        [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(16)]
    ]
    # Ensure distinct for random init
    # We'll handle distinctness in the search loop if needed, but for init let's just make sure
    # The search loop handles distinctness.

    for init in inits:
        if time.time() > deadline:
            break
        current_points = [p[:] for p in init]
        # Ensure distinct
        pts_set = set(tuple(p) for p in current_points)
        if len(pts_set) < 16:
             while len(pts_set) < 16:
                 idx = random.randint(0, 15)
                 new_p = [random.randint(-20, 20), random.randint(-20, 20)]
                 current_points[idx] = new_p
                 pts_set = set(tuple(p) for p in current_points)
        
        current_valid, current_score = count_lines(current_points)
        if not current_valid:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Local search from this init
        while time.time() < deadline:
            improved = False
            for _ in range(2000):
                if time.time() > deadline:
                    break
                idx = random.randint(0, 15)
                old_x, old_y = current_points[idx]
                
                # Random move
                # Mix of small and large moves
                if random.random() < 0.8:
                    new_x = old_x + random.randint(-5, 5)
                    new_y = old_y + random.randint(-5, 5)
                else:
                    new_x = random.randint(-50, 50)
                    new_y = random.randint(-50, 50)
                    
                new_x = max(-1000, min(1000, new_x))
                new_y = max(-1000, min(1000, new_y))
                
                if new_x == old_x and new_y == old_y:
                    continue
                
                # Check distinct
                distinct = True
                for k in range(16):
                    if k == idx: continue
                    if current_points[k][0] == new_x and current_points[k][1] == new_y:
                        distinct = False
                        break
                if not distinct:
                    continue
                
                current_points[idx] = [new_x, new_y]
                new_valid, new_score = count_lines(current_points)
                
                if new_valid and new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_points[idx] = [old_x, old_y]
                    
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Stuck, break to next init or restart
                break

    return [[p[0], p[1]] for p in best_points]
