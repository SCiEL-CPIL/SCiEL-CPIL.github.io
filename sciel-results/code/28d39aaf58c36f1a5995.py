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

    lines = defaultdict(int)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Line equation: A*x + B*y = C
            # Normalize (A, B, C)
            if dx == 0:
                A, B, C = 1, 0, x1
            elif dy == 0:
                A, B, C = 0, 1, y1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                A = dy_n
                B = -dx_n
                C = A*x1 + B*y1
                
                g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g_line == 0:
                    g_line = 1
                A //= g_line
                B //= g_line
                C //= g_line
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
            
            lines[(A, B, C)] += 1
            
    count_4 = 0
    for key, cnt in lines.items():
        if cnt > 4:
            return -1
        if cnt == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Initial 4x4 grid (Finite Field AG(2,4) structure mapped to integers)
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_4_lines(points)
    best_points = [p[:] for p in points]
    
    import time
    start_time = time.time()
    timeout = 110
    
    # Multi-start Simulated Annealing
    for restart in range(20):
        if (time.time() - start_time) > timeout:
            break
            
        # Initialize
        if restart == 0:
            points = [[x, y] for x in range(4) for y in range(4)]
            current_score = best_score
        else:
            # Perturb best points
            points = [p[:] for p in best_points]
            for i in range(16):
                if random.random() < 0.5:
                    points[i][0] += random.randint(-5, 5)
                    points[i][1] += random.randint(-5, 5)
            current_score = count_4_lines(points)
            if current_score < 0:
                continue
        
        T = 5.0
        T_min = 0.1
        cooling = 0.95
        
        while T > T_min and (time.time() - start_time) < timeout:
            i = random.randint(0, 15)
            # Larger steps for better exploration
            delta_x = random.randint(-5, 5)
            delta_y = random.randint(-5, 5)
            
            new_points = [p[:] for p in points]
            new_points[i][0] += delta_x
            new_points[i][1] += delta_y
            
            new_score = count_4_lines(new_points)
            
            if new_score < 0:
                continue
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    
            T *= cooling
            
    # Final check
    if count_4_lines(best_points) < 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
