import time
import random
import math

def count_ordinary(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize the line equation Ax + By + C = 0
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
            
            # Ensure consistent sign for the key
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
            
    return ordinary

def are_all_collinear(points):
    """
    Checks if all points lie on a single line.
    """
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    for i in range(2, len(points)):
        x, y = points[i]
        if A * x + B * y + C != 0:
            return False
    return True

def get_integer_points_on_line(p1, p2, k_range):
    """
    Returns a list of integer points on the line passing through p1 and p2,
    parameterized by k in k_range.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    g = gcd(dx, dy)
    if g == 0:
        return []
        
    dx_step = dx // g
    dy_step = dy // g
    
    points = []
    for k in k_range:
        x = x1 + k * dx_step
        y = y1 + k * dy_step
        # Bounds check
        if abs(x) <= 10**9 and abs(y) <= 10**9:
            points.append([x, y])
    return points

def local_search(points, time_limit):
    """
    Simulated annealing local search to minimize ordinary lines.
    """
    current_points = [p[:] for p in points]
    current_score = count_ordinary(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    start_time = time.time()
    T = 10.0  # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 11)
        old_point = current_points[idx][:]
        
        new_point = None
        
        # Strategy: 50% Random, 50% Structured (Collinear)
        if random.random() < 0.5:
            # Random move
            if random.random() < 0.7:
                # Local perturbation
                new_x = old_point[0] + random.randint(-10, 10)
                new_y = old_point[1] + random.randint(-10, 10)
            else:
                # Global jump
                new_x = random.randint(-10**6, 10**6)
                new_y = random.randint(-10**6, 10**6)
            new_point = [new_x, new_y]
        else:
            # Structured move: Move to be collinear with two other points
            # Pick two other points
            other_indices = [i for i in range(12) if i != idx]
            j, k = random.sample(other_indices, 2)
            p_j = current_points[j]
            p_k = current_points[k]
            
            # Get integer points on line p_j p_k
            # Try k values around 0 to find points near the segment or extensions
            candidates = get_integer_points_on_line(p_j, p_k, range(-20, 21))
            if candidates:
                # Filter out points that are already in the set (excluding idx)
                valid_candidates = [p for p in candidates if p != current_points[idx]]
                # Also ensure it's not the same as any other point
                # This is a bit expensive, but N is small
                # We can just pick one and check later
                if valid_candidates:
                    new_point = random.choice(valid_candidates)
                    
        if new_point is None:
            continue
            
        # Check distinctness
        if any(current_points[i] == new_point for i in range(12) if i != idx):
            continue
            
        # Check bounds
        if abs(new_point[0]) > 10**9 or abs(new_point[1]) > 10**9:
            continue
            
        current_points[idx] = new_point
        new_score = count_ordinary(current_points)
        
        # Check if all collinear (invalid)
        if are_all_collinear(current_points):
            # Revert immediately
            current_points[idx] = old_point
            continue
            
        delta = new_score - current_score
        
        if delta < 0:
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                current_points[idx] = old_point
                
        # Cool down
        T *= cooling_rate
        
    return best_points, best_score

def construct():
    # Deadline guard
    deadline = time.time() + 110  # 110s to be safe
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    configs = []
    
    # 3x4 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Random points
    for _ in range(5):
        configs.append([[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(12)])
        
    # Points on a parabola
    configs.append([[x, x*x] for x in range(12)])
    
    # Points on a circle
    configs.append([[int(100*math.cos(2*math.pi*i/12)), int(100*math.sin(2*math.pi*i/12))] for i in range(12)])
    
    # Run local search on each config
    num_configs = len(configs)
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Ensure distinct points
        if len(set(tuple(p) for p in config)) < 12:
            continue
            
        # Allocate time
        time_remaining = deadline - time.time()
        time_per_config = max(1, time_remaining / (num_configs - i))
        
        points, score = local_search(config, time_per_config)
        
        # Validate
        if len(set(tuple(p) for p in points)) < 12:
            continue
        if are_all_collinear(points):
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
            continue
            
        if score < best_score:
            best_score = score
            best_points = points
            
    # If best_points is None or not valid, return a default
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
