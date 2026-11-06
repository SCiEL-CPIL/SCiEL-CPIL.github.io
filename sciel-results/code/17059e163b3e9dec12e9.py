import random
import time
import math
from itertools import combinations

def _get_line_key(p1, p2):
    """
    Returns a unique key for the line passing through p1 and p2.
    The key is normalized such that (dx, dy, c) is unique for the line.
    We represent the line as Ax + By + C = 0.
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    We normalize by dividing by gcd(A, B, C) and ensuring a canonical sign.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    if A == 0 and B == 0:
        return None # Same point, shouldn't happen for distinct points
        
    # Normalize A, B, C
    # Find GCD of A, B, C
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a
        
    g = gcd(gcd(A, B), C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    # Canonical sign: First non-zero of A, B should be positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate(points):
    """
    Evaluates a set of 16 points.
    Returns (score, is_valid).
    score: number of lines containing exactly 4 points.
    is_valid: True if no line contains >= 5 points and all points are distinct.
    """
    n = len(points)
    if n != 16:
        return 0, False
    
    # Check distinctness and bounds
    seen = set()
    for p in points:
        x, y = p
        if (x, y) in seen:
            return 0, False
        seen.add((x, y))
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
            
    line_counts = {}
    
    # Iterate over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            key = _get_line_key(p1, p2)
            if key is None: continue
            
            if key not in line_counts:
                line_counts[key] = set([i, j])
            else:
                line_counts[key].add(i)
                line_counts[key].add(j)
                
    score = 0
    for key, pts_in_line in line_counts.items():
        count = len(pts_in_line)
        if count == 4:
            score += 1
        elif count >= 5:
            return score, False # Invalid configuration
            
    return score, True

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    # This gives a baseline score.
    # Grid points: (0,0) to (3,3)
    # Let's shift to center it or just use small integers.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score, best_valid = _evaluate(best_points)
    
    if not best_valid:
        # If initial is invalid (it shouldn't be), try random
        best_points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
        best_score, best_valid = _evaluate(best_points)
        
    current_points = [p[:] for p in best_points]
    current_score, current_valid = _evaluate(current_points)
    
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.1
    
    # We will perform moves until time runs out
    while time.time() - start_time < timeout:
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random move: change x or y by a small amount, or jump
        # Strategy: mix of local steps and occasional jumps
        if random.random() < 0.8:
            # Local step
            step = random.randint(-2, 2)
            if random.random() < 0.5:
                new_x = old_x + step
                new_y = old_y
            else:
                new_x = old_x
                new_y = old_y + step
        else:
            # Jump within a reasonable range
            new_x = old_x + random.randint(-50, 50)
            new_y = old_y + random.randint(-50, 50)
            
        # Keep within bounds
        if new_x < -1000: new_x = -1000
        if new_x > 1000: new_x = 1000
        if new_y < -1000: new_y = -1000
        if new_y > 1000: new_y = 1000
        
        # Apply move
        current_points[idx] = [new_x, new_y]
        
        # Evaluate
        new_score, new_valid = _evaluate(current_points)
        
        if new_valid:
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = current_score - new_score
                if delta == 0:
                    # Same score, accept with some probability to explore
                    if random.random() < 0.5:
                        current_score = new_score
                    else:
                        current_points[idx] = [old_x, old_y]
                else:
                    if random.random() < math.exp(-delta / temperature):
                        current_score = new_score
                    else:
                        current_points[idx] = [old_x, old_y]
        else:
            # Invalid (5+ collinear or duplicate)
            # Reject strongly, but maybe accept if temperature is very high to escape?
            # For now, reject invalid moves immediately to maintain constraint robustness
            # Or accept with very low probability
            if random.random() < math.exp(-5.0 / temperature):
                 current_score = new_score # Penalty score effectively
            else:
                 current_points[idx] = [old_x, old_y]
                 
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    score, valid = _evaluate(pts)
    print(f"Score: {score}, Valid: {valid}")
    print(pts)
