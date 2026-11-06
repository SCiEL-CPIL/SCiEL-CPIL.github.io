import math
import random
import itertools

def count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    # Use a set of normalized line representations
    # A line can be represented by (A, B, C) for Ax + By + C = 0
    # Normalization: gcd(A, B, C) = 1, A > 0 or (A == 0 and B > 0)
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            # Find gcd of A, B, C
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical form
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    # Count lines with exactly 4 points
    count = 0
    for key, pts in line_counts.items():
        if len(pts) == 4:
            count += 1
            
    return count

def is_valid(points):
    """
    Check if points are distinct and no 5 are collinear.
    """
    if len(set(map(tuple, points))) != len(points):
        return False
        
    # Check for 5 collinear
    # Reuse line logic
    n = len(points)
    line_counts = {}
    import math
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            return False
            
    return True

def construct():
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = count_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    import time
    start_time = time.time()
    time_limit = 110
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    # Track stagnation
    stagnation_count = 0
    max_stagnation = 5000
    
    while time.time() - start_time < time_limit:
        # Occasionally restart temperature or point to escape local minima
        if stagnation_count > max_stagnation:
            # Reheat
            T = max(T, 5.0)
            stagnation_count = 0
            
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Move strategy: 90% small, 10% large
        if random.random() < 0.1:
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
        else:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
        # Bounds
        if new_x < -1000: new_x = -1000
        if new_x > 1000: new_x = 1000
        if new_y < -1000: new_y = -1000
        if new_y > 1000: new_y = 1000
        
        points[idx] = [new_x, new_y]
        
        if not is_valid(points):
            points[idx] = [old_x, old_y]
            continue
            
        new_score = count_lines(points)
        
        if new_score > current_score:
            current_score = new_score
            stagnation_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() > prob:
                    points[idx] = [old_x, old_y]
                    stagnation_count += 1
            else:
                stagnation_count += 1 # No improvement
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_points
