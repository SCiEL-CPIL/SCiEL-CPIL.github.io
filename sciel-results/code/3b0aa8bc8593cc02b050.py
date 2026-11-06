import random
import math
from collections import defaultdict
import time

def count_4_lines(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check distinctness
    if len(set(map(tuple, points))) != 16:
        return -1
        
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1

    lines = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the line equation A*x + B*y = C
            A = dy
            B = -dx
            C = A*x1 + B*y1
            
            if A == 0 and B == 0:
                continue
            
            g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_line == 0:
                g_line = 1
            A_n = A // g_line
            B_n = B // g_line
            C_n = C // g_line
            
            if A_n < 0 or (A_n == 0 and B_n < 0):
                A_n = -A_n
                B_n = -B_n
                C_n = -C_n
                
            key = (A_n, B_n, C_n)
            lines[key].append(i)
            
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) > 4:
            return -1
        if len(pts) == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_4_lines(points)
    best_points = [p[:] for p in points]
    
    # Simulated Annealing with improved moves
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    import time
    start_time = time.time()
    time_limit = 110
    
    # Random seed for reproducibility in testing, but random is fine
    random.seed(42)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Strategy: 
        # 1. Move one point (small step)
        # 2. Move two points (coupled)
        # 3. Random restart (rare)
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Move one point
            i = random.randint(0, 15)
            # Small steps to stay near the "structure"
            delta_x = random.randint(-3, 3)
            delta_y = random.randint(-3, 3)
            
            new_points = [p[:] for p in points]
            new_points[i][0] += delta_x
            new_points[i][1] += delta_y
            
        elif move_type < 0.9:
            # Move two points
            i, j = random.sample(range(16), 2)
            delta_x1 = random.randint(-3, 3)
            delta_y1 = random.randint(-3, 3)
            delta_x2 = random.randint(-3, 3)
            delta_y2 = random.randint(-3, 3)
            
            new_points = [p[:] for p in points]
            new_points[i][0] += delta_x1
            new_points[i][1] += delta_y1
            new_points[j][0] += delta_x2
            new_points[j][1] += delta_y2
        else:
            # Large jump / restart
            # Perturb a few points significantly
            idxs = random.sample(range(16), 3)
            new_points = [p[:] for p in points]
            for i in idxs:
                new_points[i][0] += random.randint(-10, 10)
                new_points[i][1] += random.randint(-10, 10)
        
        new_score = count_4_lines(new_points)
        
        if new_score < 0:
            continue
            
        delta = new_score - best_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                
        T *= cooling
        
    # Final Hill Climbing
    # Try to improve by moving single points to best local position
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(16):
            original_x, original_y = best_points[i]
            best_local_score = best_score
            best_local_point = [original_x, original_y]
            
            # Search a small neighborhood for point i
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx = original_x + dx
                    ny = original_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [nx, ny]
                    score = count_4_lines(candidate)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_point = [nx, ny]
            
            if best_local_score > best_score:
                best_points[i] = best_local_point
                best_score = best_local_score
                improved = True
                
    # Final check
    if count_4_lines(best_points) < 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
