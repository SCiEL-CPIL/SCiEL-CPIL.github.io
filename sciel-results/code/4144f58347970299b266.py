import random
import time
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_key = (A, B, C)
            
            if line_key not in line_counts:
                line_counts[line_key] = 0
            line_counts[line_key] += 1
            
    ordinary_count = 0
    for count_pairs in line_counts.values():
        if count_pairs == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_points = None
    best_score = float('inf')
    
    def try_update(points, score):
        nonlocal best_points, best_score
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            return best_score == 6
    
    def local_search(points, max_iters=5000):
        nonlocal best_score, best_points
        current_points = [p[:] for p in points]
        current_score = count_ordinary_lines(current_points)
        try_update(current_points, current_score)
        if best_score == 6:
            return
        
        no_improvement = 0
        max_no_improvement = 3000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if no_improvement >= max_no_improvement:
                break
            
            idx = random.randint(0, 11)
            old_p = current_points[idx]
            
            # Try larger moves to escape local optima
            scale = random.choice([1, 2, 3, 5, 10, 20])
            new_x = old_p[0] + random.randint(-scale, scale)
            new_y = old_p[1] + random.randint(-scale, scale)
            
            if new_x == old_p[0] and new_y == old_p[1]:
                continue
            
            # Check distinctness and bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improvement += 1
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = [new_x, new_y]
            
            new_score = count_ordinary_lines(candidate_points)
            
            if new_score < current_score:
                current_points = candidate_points
                current_score = new_score
                no_improvement = 0
                try_update(current_points, current_score)
                if best_score == 6:
                    return
            else:
                no_improvement += 1
                
    # Generate diverse initial configurations
    inits = []
    
    # 1. 4x3 grid
    inits.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 3x4 grid
    inits.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Near-pencil: 11 points on a line, 1 off
    inits.append([[i, 0] for i in range(11)] + [[5, 1]])
    
    # 4. Two lines with 6 points each (sharing no points)
    inits.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(6)])
    
    # 5. Three lines with 4 points each
    inits.append([[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)])
    
    # 6. Random configurations
    for _ in range(5):
        pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(12)]
        if len(set(tuple(p) for p in pts)) == 12:
            inits.append(pts)
    
    # 7. Based on known good configurations - try to approximate Hesse configuration
    # Hesse has 9 inflection points on a cubic, but we need 12 points
    # Try arrangements with many collinear triples
    
    for init_points in inits:
        if time.time() > deadline:
            break
            
        # Ensure distinct
        if len(set(tuple(p) for p in init_points)) != 12:
            continue
            
        local_search(init_points, max_iters=3000)
        
        if best_score == 6:
            break
    
    # If we still haven't found a good solution, try more aggressive random restarts
    if best_score > 10 and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            
            pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(12)]
            if len(set(tuple(p) for p in pts)) == 12:
                local_search(pts, max_iters=2000)
                if best_score == 6:
                    break
    
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
