import itertools
import random
import time

def count_ordinary_lines(points):
    if len(points) < 2:
        return 0
    
    # Map from line representation to set of points on that line
    # Line representation: (A, B, C) where Ax + By + C = 0
    # Normalize A, B, C to be coprime and A > 0 (or A==0, B>0)
    
    lines = {}
    for p1, p2 in itertools.combinations(points, 2):
        x1, y1 = p1
        x2, y2 = p2
        
        # Line coefficients
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize
        if A == 0 and B == 0:
            continue # Duplicate points, shouldn't happen
            
        # Find GCD
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        # Ensure canonical form
        if A < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0 and B < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0 and B == 0 and C < 0:
            C = -C
            
        key = (A, B, C)
        
        if key not in lines:
            lines[key] = set()
            
        lines[key].add(p1)
        lines[key].add(p2)
        
    ordinary_count = 0
    for line_key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Baseline: 3x4 grid
    base_points = [[x, y] for x in range(4) for y in range(3)]
    best_points = [p[:] for p in base_points]
    best_score = count_ordinary_lines(best_points)
    
    # If we have time, try to improve
    # Deadline: 110 seconds (safety margin)
    start_time = time.time()
    deadline = start_time + 110
    
    # Local search / Random restarts
    # We can try moving points to nearby integer coordinates
    # Or swapping points
    
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Create a perturbed copy
        current_points = [p[:] for p in best_points]
        
        # Randomly select 1-3 points to move
        num_moves = random.randint(1, 3)
        indices = random.sample(range(12), num_moves)
        
        for idx in indices:
            # Move point to a random nearby location
            # Keep within bounds
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            # Ensure distinctness
            new_point = [new_x, new_y]
            if new_point in current_points:
                continue
                
            current_points[idx] = new_point
            
        # Check if all distinct
        if len(set(tuple(p) for p in current_points)) != 12:
            continue
            
        # Check if not all collinear
        # (Simple check: if all points lie on a line, score is infinite/invalid, but problem says not all collinear)
        # We assume perturbations won't make them all collinear easily, but let's be safe.
        # Actually, if they become collinear, ordinary lines = 0? No, 1 line with 12 points.
        # But problem requires "not all collinear".
        # Let's check collinearity.
        is_collinear = True
        for i in range(1, 11):
            if (current_points[i][0] - current_points[0][0]) * (current_points[11][1] - current_points[0][1]) != \
               (current_points[i][1] - current_points[0][1]) * (current_points[11][0] - current_points[0][0]):
                is_collinear = False
                break
        if is_collinear:
            continue
            
        score = count_ordinary_lines(current_points)
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in current_points]
            
            # If we reach the theoretical minimum 6, we can stop
            if best_score <= 6:
                break
                
    return best_points
