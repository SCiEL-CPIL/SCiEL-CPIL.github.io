import random
import time
import math
from itertools import combinations

def _line_key(p1, p2):
    """
    Returns a canonical representation of the line passing through p1 and p2.
    Line: ax + by = c
    a = y2 - y1
    b = x1 - x2
    c = a*x1 + b*y1
    Normalize (a, b, c) by gcd and sign.
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    
    # Normalize
    if a == 0 and b == 0:
        return None # Should not happen for distinct points
        
    # GCD of a, b, c
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
        
    # Canonical sign: first non-zero of (a, b) should be positive
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def evaluate(points):
    """
    Evaluates a set of 16 points.
    Returns (score, is_valid).
    score: number of lines containing exactly 4 points.
    is_valid: False if any line contains 5 or more points.
    """
    if len(points) != 16:
        return 0, False
        
    # Check distinctness
    if len(set(map(tuple, points))) != 16:
        return 0, False

    line_map = {}
    # Iterate over all pairs to build line map
    for i in range(16):
        for j in range(i + 1, 16):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    score = 0
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return 0, False
        if len(idx_set) == 4:
            score += 1
            
    return score, True

def construct():
    # Initial best: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = evaluate(best_points)
    
    # Time management
    start_time = time.time()
    timeout = 110  # 10s buffer for overhead
    
    current_points = [p[:] for p in best_points]
    current_score, _ = evaluate(current_points)
    
    # Helper to generate a random neighbor
    def get_neighbor(pts):
        # 50% swap two points, 50% move one point
        if random.random() < 0.5:
            i, j = random.sample(range(16), 2)
            new_pts = [p[:] for p in pts]
            new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
            return new_pts
        else:
            i = random.randint(0, 15)
            # Move to a random nearby integer
            # Keep coordinates within [-100, 100]
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_pts = [p[:] for p in pts]
            new_pts[i][0] += dx
            new_pts[i][1] += dy
            # Clamp to bounds
            if new_pts[i][0] > 1000: new_pts[i][0] = 1000
            if new_pts[i][0] < -1000: new_pts[i][0] = -1000
            if new_pts[i][1] > 1000: new_pts[i][1] = 1000
            if new_pts[i][1] < -1000: new_pts[i][1] = -1000
            return new_pts

    # Simulated Annealing / Hill Climbing
    # We use a simple greedy approach with random restarts if stuck
    # Or just a long run. Given the small state space, greedy might work well.
    
    # To be safe, we'll do a mixed approach:
    # 1. Try to improve current.
    # 2. If stuck for X steps, do a random jump.
    
    steps_since_improvement = 0
    max_steps_no_improve = 1000
    
    while time.time() - start_time < timeout:
        new_points = get_neighbor(current_points)
        new_score, is_valid = evaluate(new_points)
        
        if is_valid:
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                steps_since_improvement = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                steps_since_improvement += 1
                
        if steps_since_improvement > max_steps_no_improve:
            # Random restart from a perturbed grid or random valid set
            # Let's restart from a random perturbation of the best known
            current_points = [p[:] for p in best_points]
            # Apply a few random moves
            for _ in range(10):
                current_points = get_neighbor(current_points)
            current_score, is_valid = evaluate(current_points)
            if not is_valid:
                # Revert to best
                current_points = [p[:] for p in best_points]
                current_score = best_score
            steps_since_improvement = 0
            
    return best_points
