import time
import random
import math
from collections import defaultdict

def _canonical_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
        
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def _compute_line_map(points):
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _score_from_line_map(line_points):
    score = 0
    for pts in line_points.values():
        if len(pts) == 4:
            score += 10
        elif len(pts) >= 5:
            score -= 1000
    return score

def _score(points):
    line_points = _compute_line_map(points)
    return _score_from_line_map(line_points)

def _initial_grid():
    # 4x4 grid, scaled to use space but stay well within bounds
    # Step size 100 allows for large perturbations if needed
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i * 100, j * 100])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the known best: 4x4 grid (10 lines)
    points = _initial_grid()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    current_score = best_score
    T = 10.0
    min_T = 0.1
    cooling = 0.999
    
    step = 0
    stagnant_steps = 0
    
    while time.time() < deadline:
        step += 1
        
        # Adaptive move size
        # If we've been stuck, increase move size
        move_range = 50
        if stagnant_steps > 5000:
            move_range = 200
        elif stagnant_steps > 10000:
            move_range = 500
            
        # Choose move type
        if random.random() < 0.8:
            # Single point move
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Random new position
            new_x = old_x + random.randint(-move_range, move_range)
            new_y = old_y + random.randint(-move_range, move_range)
            
            # Clamp to bounds
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            if any(new_x == p[0] and new_y == p[1] for i, p in enumerate(points) if i != idx):
                continue
            
            points[idx] = [new_x, new_y]
            new_score = _score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                stagnant_steps = 0
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    points[idx] = [old_x, old_y]
                    stagnant_steps += 1
            if new_score >= best_score:
                stagnant_steps = 0
            else:
                stagnant_steps += 1
                
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            old_i = points[i][:]
            old_j = points[j][:]
            points[i], points[j] = points[j], points[i]
            
            new_score = _score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                stagnant_steps = 0
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    points[i], points[j] = old_i, old_j
                    stagnant_steps += 1
            if new_score >= best_score:
                stagnant_steps = 0
            else:
                stagnant_steps += 1

        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Periodic restart if very stuck
        if stagnant_steps > 20000 and time.time() < deadline - 10:
            # Try a random perturbation of the best solution
            points = [p[:] for p in best_points]
            current_score = best_score
            # Perturb 2 random points significantly
            for _ in range(2):
                idx = random.randint(0, 15)
                points[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            current_score = _score(points)
            stagnant_steps = 0
            T = 5.0

    return best_points
