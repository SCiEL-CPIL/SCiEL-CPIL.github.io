import random
import itertools
import math
import time

def count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
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

def get_penalty(points):
    """
    Returns the number of lines with >= 5 points.
    """
    n = len(points)
    line_counts = {}
    
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
            
    penalty = 0
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            penalty += 1
            
    return penalty

def is_valid(points):
    """
    Check if points are distinct and no 5 are collinear.
    """
    if len(set(map(tuple, points))) != len(points):
        return False
        
    return get_penalty(points) == 0

def construct():
    import time
    start_time = time.time()
    limit = 110 # seconds
    
    # Start with a random valid configuration or a grid
    # Grid is a good start because it has many 4-point lines
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Ensure distinctness (grid is distinct)
    # Check validity
    if not is_valid(points):
        # If grid is invalid (it shouldn't be, 4x4 grid has no 5 collinear? 
        # Wait, in a 4x4 grid, max collinear is 4. So it is valid.)
        pass
        
    best_score = count_lines(points)
    best_points = [p[:] for p in points]
    current_score = best_score
    current_penalty = 0
    
    # Simulated Annealing
    # Score = count_lines - 1000 * penalty
    # We want to maximize this.
    
    T = 100.0 # Initial temperature
    alpha = 0.995 # Cooling rate
    
    while time.time() - start_time < limit:
        # Generate a neighbor
        # Move a random point to a random location
        # Strategy: 50% small move, 50% large move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        if random.random() < 0.5:
            # Small move
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            # Large move
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        new_points = [p[:] for p in points]
        new_points[idx] = [new_x, new_y]
        
        # Check if distinct
        if len(set(map(tuple, new_points))) != 16:
            continue
            
        # Calculate new metrics
        new_penalty = get_penalty(new_points)
        new_count = count_lines(new_points)
        
        new_score = new_count - 1000 * new_penalty
        current_score_val = current_score - 1000 * current_penalty
        
        # Acceptance criterion
        if new_score > current_score_val:
            # Better solution
            points = new_points
            current_score = new_count
            current_penalty = new_penalty
            
            if new_penalty == 0 and new_count > best_score:
                best_score = new_count
                best_points = [p[:] for p in points]
        else:
            # Worse or equal
            delta = current_score_val - new_score
            if delta < 0:
                delta = 0
            # Metropolis: accept with probability exp(-delta / T)
            # Since we are maximizing, if new < current, delta = current - new > 0
            # We accept if random < exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                points = new_points
                current_score = new_count
                current_penalty = new_penalty
                if new_penalty == 0 and new_count > best_score:
                    best_score = new_count
                    best_points = [p[:] for p in points]
                    
        # Cool down
        T *= alpha
        
    return best_points
