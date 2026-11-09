import time
import random
import math
from collections import defaultdict

def _canonical_line(p1, p2):
    """
    Returns a canonical representation of the line passing through p1 and p2.
    p1, p2 are tuples (x, y).
    The line is defined by ax + by + c = 0.
    We normalize (a, b, c) such that:
    1. gcd(|a|, |b|, |c|) = 1
    2. a > 0, or a == 0 and b > 0, or a == 0 and b == 0 and c > 0
    """
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
    """Compute line -> set of point indices mapping."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _score_from_line_map(line_points):
    """Score from a precomputed line map."""
    score = 0
    for pts in line_points.values():
        if len(pts) == 4:
            score += 10
        elif len(pts) >= 5:
            score -= 100
    return score

def _score(points):
    """Calculate the score for a set of 16 points."""
    line_points = _compute_line_map(points)
    return _score_from_line_map(line_points)

def _initial_seed():
    """Generate a structured initial seed based on a perturbed 4x4 grid."""
    random.seed(42)
    # Start with a 4x4 grid, then perturb each point slightly
    base_points = []
    for i in range(4):
        for j in range(4):
            base_points.append([i * 2, j * 2])
    
    # Perturb each point to break exact grid structure while keeping many 4-lines
    points = []
    for p in base_points:
        x = p[0] + random.randint(-1, 1)
        y = p[1] + random.randint(-1, 1)
        # Ensure distinctness
        while any(x == q[0] and y == q[1] for q in points):
            x = p[0] + random.randint(-2, 2)
            y = p[1] + random.randint(-2, 2)
        points.append([x, y])
    
    # Scale to use more of the coordinate space
    scale = random.randint(3, 8)
    points = [[x * scale, y * scale] for x, y in points]
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Use structured seed
    points = _initial_seed()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    # Precompute line map for incremental updates
    line_map = _compute_line_map(points)
    current_score = _score_from_line_map(line_map)
    
    # Simulated Annealing with mixed moves
    T = 50.0
    min_T = 0.1
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose move type: 70% single point move, 30% 2-opt swap
        if random.random() < 0.7:
            # Single point move
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            if random.random() < 0.5:
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            new_point = [new_x, new_y]
            if any(new_point == p for i, p in enumerate(points) if i != idx):
                continue
            
            # Apply move
            points[idx] = new_point
            
            # Recompute score (simple but correct; incremental would be faster)
            new_score = _score(points)
            
            if new_score >= best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                current_score = new_score
                line_map = _compute_line_map(points)
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                    line_map = _compute_line_map(points)
                else:
                    points[idx] = [old_x, old_y]
        else:
            # 2-opt swap: swap two points
            i, j = random.sample(range(16), 2)
            old_i = points[i][:]
            old_j = points[j][:]
            points[i], points[j] = points[j], points[i]
            
            new_score = _score(points)
            
            if new_score >= best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                current_score = new_score
                line_map = _compute_line_map(points)
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                    line_map = _compute_line_map(points)
                else:
                    points[i], points[j] = old_i, old_j
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Periodic restart if stuck
        if step % 5000 == 0 and best_score < 15:
            # Try a different structured seed
            alt_seed = _initial_seed()
            alt_score = _score(alt_seed)
            if alt_score > best_score:
                points = [p[:] for p in alt_seed]
                best_points = [p[:] for p in alt_seed]
                best_score = alt_score
                current_score = alt_score
                line_map = _compute_line_map(points)
    
    # Final polish: local search around best
    if time.time() < deadline - 10:
        polish_points = [p[:] for p in best_points]
        polish_score = best_score
        polish_line_map = _compute_line_map(polish_points)
        
        polish_T = 5.0
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, 15)
            old_x, old_y = polish_points[idx]
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            new_point = [new_x, new_y]
            if any(new_point == p for i, p in enumerate(polish_points) if i != idx):
                continue
            
            polish_points[idx] = new_point
            new_score = _score(polish_points)
            
            if new_score >= polish_score:
                polish_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in polish_points]
            else:
                delta = new_score - polish_score
                if random.random() < math.exp(delta / polish_T):
                    polish_score = new_score
                else:
                    polish_points[idx] = [old_x, old_y]
            
            polish_T *= 0.99
            if polish_T < 0.5:
                polish_T = 0.5
    
    return best_points
