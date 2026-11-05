import random
import time
import math
from math import gcd

def normalize_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    
    if A == 0 and B == 0:
        return (0, 0, 0)
        
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
            
    return (A, B, C)

def count_ordinary_lines(points):
    line_points = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = normalize_line(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    # Check if all collinear
    if len(line_points) == 1:
        pts = list(line_points.values())[0]
        if len(pts) == n:
            return float('inf')
            
    ordinary_count = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial candidates
    grid_34 = [[x, y] for x in range(4) for y in range(3)]
    best_points = [tuple(p) for p in grid_34]
    best_score = count_ordinary_lines(best_points)
    
    # Try a few other structured starts
    # 2x6 grid
    grid_26 = [[x, y] for x in range(2) for y in range(6)]
    score_26 = count_ordinary_lines([tuple(p) for p in grid_26])
    if score_26 < best_score:
        best_score = score_26
        best_points = [tuple(p) for p in grid_26]
        
    # Random starts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            current_points = [tuple(p) for p in grid_34]
        else:
            current_points = [(random.randint(-5, 5), random.randint(-5, 5)) for _ in range(12)]
            while len(set(current_points)) < 12:
                idx = random.randint(0, 11)
                current_points[idx] = (random.randint(-5, 5), random.randint(-5, 5))
        
        current_score = count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = current_points[:]
        
        # Local Search / SA
        T = 5.0
        cooling = 0.99
        max_steps = 20000
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Local move
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            # Occasionally jump
            if random.random() < 0.05:
                new_x = random.randint(-10, 10)
                new_y = random.randint(-10, 10)
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            new_point = (new_x, new_y)
            
            # Check distinct
            is_distinct = True
            for k in range(12):
                if k != idx and current_points[k] == new_point:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            old_point = current_points[idx]
            current_points[idx] = new_point
            
            new_score = count_ordinary_lines(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
            else:
                delta = new_score - current_score
                if delta == 0:
                    current_score = new_score
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < math.exp(-delta / T):
                        current_score = new_score
                    else:
                        current_points[idx] = old_point
            
            T *= cooling
            if T < 0.1:
                T = 0.1

    return [list(p) for p in best_points]
