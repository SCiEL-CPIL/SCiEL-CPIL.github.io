"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _score(points):
    """
    Calculate the score for a set of 16 points.
    Maximize lines with exactly 4 points.
    Penalize lines with >= 5 points.
    """
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
            
    score = 0
    for pts in line_points.values():
        if len(pts) == 4:
            score += 10
        elif len(pts) >= 5:
            score -= 100
            
    return score

def _evaluate_raw(points):
    """
    Returns the raw value: number of lines with exactly 4 points,
    and a validity flag (True if no line has >= 5 points).
    """
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    valid = True
    for pts in line_points.values():
        k = len(pts)
        if k == 4:
            count_4 += 1
        elif k >= 5:
            valid = False
            
    return count_4, valid

def _local_search(points, time_limit, seed=None):
    """
    Perform local search / simulated annealing starting from 'points'.
    Returns the best point configuration found.
    """
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    deadline = start_time + time_limit
    
    pts = [p[:] for p in points]
    current_score = _score(pts)
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    T = 20.0
    min_T = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = pts[idx]
        
        # Move strategy: mix of small perturbations and larger jumps
        # Small perturbations help fine-tune, larger jumps help escape local minima
        if random.random() < 0.7:
            # Small step
            step = random.randint(1, 5)
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)
        else:
            # Larger jump within a reasonable range
            new_x = old_x + random.randint(-20, 20)
            new_y = old_y + random.randint(-20, 20)
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        if any(new_point == p for i, p in enumerate(pts) if i != idx):
            continue
            
        # Apply move
        pts[idx] = new_point
        
        new_score = _score(pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = [p[:] for p in pts]
        elif new_score >= current_score:
            # Accept if better or equal
            current_score = new_score
        else:
            # Accept worse solution with probability exp(delta/T)
            delta = new_score - current_score
            if random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                # Reject move
                pts[idx] = [old_x, old_y]
                
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_pts_global = None
    best_raw_global = -1
    best_valid_global = False
    
    # Initial configuration: 4x4 grid
    grid_pts = [[x, y] for x in range(4) for y in range(4)]
    
    # Run multiple restarts with different initial configurations
    # Restart 1: Start from grid
    if time.time() < deadline:
        t_limit = min(30, deadline - time.time())
        res = _local_search(grid_pts, t_limit, seed=42)
        raw, valid = _evaluate_raw(res)
        if valid and raw > best_raw_global:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
        elif valid and raw == best_raw_global and not best_valid_global:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = True
        elif not best_valid_global and valid:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = True
        elif best_pts_global is None:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = valid
    
    # Restart 2-5: Random initial configurations
    for r in range(5):
        if time.time() >= deadline - 10:  # Stop if less than 10s left
            break
            
        t_limit = min(15, deadline - time.time() - 10)
        if t_limit <= 0:
            break
            
        # Generate random initial points
        rand_pts = []
        while len(rand_pts) < 16:
            p = [random.randint(-50, 50), random.randint(-50, 50)]
            if p not in rand_pts:
                rand_pts.append(p)
                
        res = _local_search(rand_pts, t_limit, seed=100 + r)
        raw, valid = _evaluate_raw(res)
        
        # Update best
        if valid and raw > best_raw_global:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = True
        elif valid and raw == best_raw_global:
            # Tie: keep existing
            pass
        elif not best_valid_global and valid:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = True
        elif best_pts_global is None:
            best_raw_global = raw
            best_pts_global = [p[:] for p in res]
            best_valid_global = valid
    
    # If we still don't have a valid solution, return the grid (which is valid)
    if best_pts_global is None or not best_valid_global:
        # The 4x4 grid is always valid (no 5 collinear)
        return grid_pts
        
    return best_pts_global
# EVOLVE-BLOCK-END
