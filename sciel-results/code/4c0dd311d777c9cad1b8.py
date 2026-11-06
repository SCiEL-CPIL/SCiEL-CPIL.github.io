import random
import time
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    if len(points) < 2:
        return 0
    
    # Dictionary to store lines. Key: (slope, intercept) or a canonical representation.
    # For integer points, we can use the line equation ax + by + c = 0 with gcd(a,b,c)=1.
    # Or simpler: for each pair, compute the line and count points on it.
    # Since N=12, N*(N-1)/2 = 66 pairs. This is small.
    
    line_counts = {}
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            
            # Calculate line coefficients
            # A(x2 - x1) + B(y2 - y1) = 0
            # A = y2 - y1
            # B = x1 - x2
            # C = -(A*x1 + B*y1)
            
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            
            # Normalize the line representation to handle equivalent lines
            # Find GCD of A, B, C
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                # Degenerate case? No, distinct points imply not all A,B,C are 0.
                # But if A=0, B!=0, C=0 etc.
                # If A=0, B=0, C=0 is impossible for distinct points.
                # If g is 0, it means A=B=C=0 which is not possible.
                pass
            
            # Ensure a canonical sign (e.g., first non-zero of A, B is positive)
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_key = (A, B, C)
            
            if line_key not in line_counts:
                line_counts[line_key] = 0
            line_counts[line_key] += 1
            
    # Now, for each line, we need to know how many distinct points are on it.
    # The count above is the number of PAIRS on the line.
    # If a line has k points, it has k*(k-1)/2 pairs.
    # We want to find lines where the number of points is exactly 2.
    # This corresponds to pair count = 1.
    
    ordinary_count = 0
    for count_pairs in line_counts.values():
        # If count_pairs == 1, then k*(k-1)/2 = 1 => k=2.
        # If count_pairs == 3, then k*(k-1)/2 = 3 => k=3.
        # If count_pairs == 6, then k*(k-1)/2 = 6 => k=4.
        # So we just check if count_pairs == 1.
        if count_pairs == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations to try
    # 1. Random
    # 2. Grid 4x3
    # 3. Grid 3x4
    
    def generate_initial():
        # 4x3 grid
        return [[x, y] for x in range(4) for y in range(3)]
        
    def generate_random():
        return [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        
    # Try a few initializations
    inits = [
        generate_initial(),
        generate_random(),
        generate_random(),
        generate_random()
    ]
    
    for init_points in inits:
        if time.time() > deadline:
            break
            
        # Ensure distinct
        if len(set(tuple(p) for p in init_points)) != 12:
            continue
            
        current_points = [p[:] for p in init_points]
        current_score = count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
        # Local Search
        no_improvement_count = 0
        max_no_improvement = 2000
        
        while time.time() < deadline and no_improvement_count < max_no_improvement:
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_p = current_points[idx]
            
            # Generate a new candidate position
            # Try moving in a small range
            new_x = old_p[0] + random.randint(-3, 3)
            new_y = old_p[1] + random.randint(-3, 3)
            
            if new_x == old_p[0] and new_y == old_p[1]:
                continue
                
            # Check distinctness
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improvement_count += 1
                continue
                
            # Calculate new score
            candidate_points = current_points.copy()
            candidate_points[idx] = [new_x, new_y]
            
            new_score = count_ordinary_lines(candidate_points)
            
            if new_score < current_score:
                current_points = candidate_points
                current_score = new_score
                no_improvement_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    
                if best_score == 6:
                    break
            else:
                no_improvement_count += 1
                
        if best_score == 6:
            break

    # If we didn't find 6, return the best found. 
    # If best_points is None (shouldn't happen), return a default.
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
