import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safe margin before 150s
    
    # Initial state: 4x4 grid
    # This configuration has 10 lines with exactly 4 points (4 rows, 4 cols, 2 main diagonals)
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = _evaluate(initial_points)
    
    # Ensure best_score is valid (no 5 collinear)
    if best_score is None:
        best_score = 0
        best_points = _random_valid_config()
        
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing
    temperature = 1.0
    cooling_rate = 0.999
    
    total_steps = 0
    steps_per_check = 100
    
    while True:
        # Check time periodically
        if total_steps % steps_per_check == 0:
            if time.time() > deadline - 5.0:
                break
        
        # Generate move
        is_swap = random.random() >= 0.7
        if not is_swap:
            idx = random.randint(0, 15)
            old_point = current_points[idx][:]
            
            if random.random() < 0.5:
                # Local perturbation
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_x = old_point[0] + dx
                new_y = old_point[1] + dy
            else:
                # Global random
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            # Check distinctness
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            current_points[idx] = [new_x, new_y]
            revert_data = (idx, old_point)
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            current_points[i], current_points[j] = current_points[j], current_points[i]
            revert_data = (i, j, old_i, old_j)

        new_score = _evaluate(current_points)
        
        if new_score is None:
            # Invalid (5+ collinear), revert
            if not is_swap:
                current_points[revert_data[0]] = revert_data[1]
            else:
                current_points[revert_data[0]] = revert_data[2]
                current_points[revert_data[1]] = revert_data[3]
            temperature *= cooling_rate
            total_steps += 1
            continue
            
        # Acceptance criteria
        delta = new_score - current_score
        if delta > 0 or (temperature > 0 and random.random() < math.exp(delta / temperature)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            if not is_swap:
                current_points[revert_data[0]] = revert_data[1]
            else:
                current_points[revert_data[0]] = revert_data[2]
                current_points[revert_data[1]] = revert_data[3]
                
        temperature *= cooling_rate
        total_steps += 1
        
    return best_points

def _evaluate(points):
    """
    Returns the number of lines with exactly 4 points, or None if any line has >= 5 points.
    """
    n = len(points)
    if n != 16:
        return None
        
    line_counts = {}
    gcd = math.gcd
    abs_ = abs
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            g = gcd(gcd(abs_(A), abs_(B)), abs_(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    four_count = 0
    for count in line_counts.values():
        if count >= 10:
            return None
        if count == 6:
            four_count += 1
            
    return four_count

def _random_valid_config():
    """Generate a random configuration with no 5 collinear."""
    while True:
        points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
        if len(set(tuple(p) for p in points)) == 16:
            if _evaluate(points) is not None:
                return points
