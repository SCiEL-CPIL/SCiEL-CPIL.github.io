import random
import math
import time

def count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    """
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
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
        
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
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
    # Initial 4x4 grid centered at 0
    # Points: (-1.5, -1.5) ... but we need integers.
    # Use 0,1,2,3 shifted to -1,0,1,2? Or just 0..3.
    # Let's use a slightly spread out grid to allow movement.
    points = [[x, y] for x in [0, 1, 2, 3] for y in [0, 1, 2, 3]]
    
    best_points = [p[:] for p in points]
    best_score = count_lines(points)
    
    # Simulated Annealing
    start_time = time.time()
    limit = 100 # seconds
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    # To improve speed, we can pre-calculate some things or just run many iterations.
    # 120s is enough for ~100k-500k iterations in Python if optimized.
    # Let's aim for 200k iterations.
    
    while time.time() - start_time < limit and T > min_T:
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Move to a random position
        # We use a larger range to escape local minima
        # The grid is 0..3, so let's search -10..15
        new_x = random.randint(-10, 15)
        new_y = random.randint(-10, 15)
        
        # Check bounds (strictly required |x|,|y| <= 1000, so -10..15 is fine)
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_score = count_lines(points)
            delta = new_score - best_score # Note: we should compare to current score, not best, for SA
            # Actually, standard SA compares to current state score.
            # Let's keep track of current score.
            
            # Re-evaluating strategy:
            # We need current_score.
            # Let's refactor slightly to track current_score.
            
            # For simplicity in this edit, I will just do Hill Climbing with restarts if stuck, 
            # or proper SA.
            # Let's do proper SA.
            
            # I need to restructure the loop to keep current_score.
            
            pass # Placeholder for logic below
            
        else:
            points[idx] = [old_x, old_y]
            
        T *= cooling_rate
        
    # The above logic was pseudocode. I will implement the actual SA loop properly in the replacement.
    
    return best_points
