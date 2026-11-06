import time
import random
from math import gcd
from itertools import combinations

EVALUATION_LIMIT = 120
SAFETY_MARGIN = 5

def _count_ordinary(points):
    """
    Count the number of ordinary lines (lines containing exactly two points).
    Optimized for n=12.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line_key to set of point indices
    # Line key: (A, B, C) normalized
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            # Canonical sign
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts_in_line in lines.values():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _is_valid(points):
    """
    Check if the points are valid:
    - Exactly 12 distinct points
    - |x|, |y| <= 10^9
    - Not all collinear
    """
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check not all collinear
    if len(points) < 3:
        return True
        
    p1, p2, p3 = points[0], points[1], points[2]
    
    # Cross product of (p2-p1) and (p3-p1)
    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if cross != 0:
        return True
        
    # If p1, p2, p3 are collinear, check if all points are on this line.
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    
    for p in points[3:]:
        x, y = p
        if A*x + B*y + C != 0:
            return True
            
    return False

def _get_initial_configs():
    """
    Generate a list of initial configurations to start the search.
    """
    configs = []
    
    # 1. 3x4 Grid
    grid_34 = [(x, y) for x in range(4) for y in range(3)]
    configs.append(grid_34)
    
    # 2. 4x3 Grid (same points, different order, but let's scale it)
    grid_43 = [(x, y) for x in range(3) for y in range(4)]
    configs.append(grid_43)
    
    # 3. "L" shape or other asymmetric?
    # Let's try a configuration with more collinearities.
    # 4 lines of 3 points?
    # This is hard to construct manually.
    
    # 4. Random configurations
    for _ in range(5):
        random_points = []
        while len(random_points) < 12:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in random_points:
                random_points.append(p)
        if _is_valid(random_points):
            configs.append(random_points)
            
    return configs

def construct():
    global BEST_POINTS, BEST_SCORE
    
    start_time = time.time()
    deadline = start_time + EVALUATION_LIMIT - SAFETY_MARGIN
    
    best_points = None
    best_score = float('inf')
    
    initial_configs = _get_initial_configs()
    
    # Evaluate initial configs
    for config in initial_configs:
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_points = list(config)
            
    # If we already have a very good score, we might return early
    # The theoretical minimum is 6.
    
    # Local Search
    # We will run hill climbing on the best config found so far.
    # To avoid getting stuck, we will do multiple restarts if time permits.
    
    current_points = list(best_points)
    current_score = best_score
    
    max_steps = 500000
    step_count = 0
    
    # Temperature for simulated annealing
    temp = 5.0
    cooling_rate = 0.99999
    
    while time.time() < deadline:
        step_count += 1
        
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = current_points[i]
        
        # Generate a new position
        # Use a step size that allows exploring the local neighborhood
        # but also allows larger jumps occasionally
        step_size = random.choice([1, 2, 5, 10, 50, 100])
        new_x = old_x + random.randint(-step_size, step_size)
        new_y = old_y + random.randint(-step_size, step_size)
        
        # Ensure bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        new_points = list(current_points)
        new_points[i] = (new_x, new_y)
        
        if not _is_valid(new_points):
            continue
            
        new_score = _count_ordinary(new_points)
        
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points
                if best_score == 6:
                    break
        else:
            # Simulated annealing: accept worse moves with probability exp(-(delta)/temp)
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta/temp):
                current_points = new_points
                current_score = new_score
                
        temp *= cooling_rate
        if temp < 0.1:
            # Restart with a new random configuration or reset to best
            # Reset to best to continue refining
            current_points = list(best_points)
            current_score = best_score
            temp = 5.0
            
            # Optional: try a new random start
            if random.random() < 0.2:
                random_points = []
                while len(random_points) < 12:
                    p = (random.randint(-1000, 1000), random.randint(-1000, 1000))
                    if p not in random_points:
                        random_points.append(p)
                if _is_valid(random_points):
                    random_score = _count_ordinary(random_points)
                    if random_score < current_score:
                        current_points = random_points
                        current_score = random_score
                        if random_score < best_score:
                            best_score = random_score
                            best_points = random_points
                            
    # Return the best points found
    return [[p[0], p[1]] for p in best_points]

if __name__ == "__main__":
    points = construct()
    print(points)
    print("Score:", _count_ordinary(points))
