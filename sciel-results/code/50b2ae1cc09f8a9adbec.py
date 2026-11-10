"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build map from line key to set of points on that line."""
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Compute score from line map: count lines with exactly 4 points."""
    score = 0
    for pts in line_map.values():
        if len(pts) == 4:
            score += 1
    return score

def _is_valid(line_map):
    """Check if no line has 5+ points."""
    for pts in line_map.values():
        if len(pts) >= 5:
            return False
    return True

def _evaluate(points):
    """
    Returns (is_valid, score).
    """
    line_map = _build_line_map(points)
    if not _is_valid(line_map):
        return False, 0
    return True, _score_from_line_map(line_map)

def _backtracking_search(points_so_far, line_map, target_n=16, candidate_pool=None, deadline=None):
    """
    Backtracking search that adds points one at a time,
    pruning if adding a point would create a line with 5+ collinear points.
    Returns (best_points, best_score).
    """
    if deadline is None:
        deadline = time.time() + 10
    
    n = len(points_so_far)
    if n == target_n:
        score = _score_from_line_map(line_map)
        return list(points_so_far), score
    
    if time.time() > deadline:
        return None, -1
    
    if candidate_pool is None:
        # Generate candidate pool from grid
        candidate_pool = []
        for x in range(-5, 6):
            for y in range(-5, 6):
                candidate_pool.append((x, y))
        random.shuffle(candidate_pool)
    
    # Try adding each candidate point
    best_points = None
    best_score = -1
    
    for candidate in candidate_pool:
        if time.time() > deadline:
            break
        
        # Check if candidate is already in points_so_far
        if candidate in points_so_far:
            continue
        
        # Check if adding candidate would create a line with 5+ points
        new_line_map = defaultdict(set)
        for key, pts in line_map.items():
            new_line_map[key] = set(pts)
        
        valid = True
        for i in range(len(points_so_far)):
            key = _line_key(points_so_far[i], candidate)
            if key is not None:
                new_line_map[key].add(i)
                new_line_map[key].add(len(points_so_far))
                if len(new_line_map[key]) >= 5:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Recurse
        new_points = points_so_far + [candidate]
        result_points, result_score = _backtracking_search(
            new_points, new_line_map, target_n, candidate_pool, deadline
        )
        
        if result_points is not None and result_score > best_score:
            best_score = result_score
            best_points = result_points
        
        # Early termination if we found a good score
        if best_score >= 10:
            break
    
    if best_points is None:
        return None, -1
    
    return best_points, best_score

def _simulated_annealing(points, deadline):
    """Simulated annealing to improve a starting configuration."""
    current = [list(p) for p in points]
    current_valid, current_score = _evaluate(current)
    if not current_valid:
        return current, 0
    
    best_points = [list(p) for p in current]
    best_score = current_score
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Try a random move
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        if random.random() < 0.8:
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        if any(p[0] == new_x and p[1] == new_y for p in current if p is not current[idx]):
            continue
        
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        
        if not cand_valid:
            cand_score = -1
        
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
        else:
            delta = cand_score - current_score
            if delta < 0:
                prob = pow(2.718281828, delta / temp)
                if random.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        temp *= cooling
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Backtracking search with different candidate pools
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Generate diverse candidate pools
        pool_size = 20 + trial * 5
        candidate_pool = set()
        for _ in range(pool_size * 10):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            candidate_pool.add((x, y))
            if len(candidate_pool) >= pool_size:
                break
        
        # Start with a small seed set
        seed_points = []
        for _ in range(4):
            p = random.choice(list(candidate_pool))
            if p not in seed_points:
                seed_points.append(p)
        
        # Build initial line map
        initial_line_map = _build_line_map(seed_points)
        
        # Backtrack
        result_points, result_score = _backtracking_search(
            seed_points, initial_line_map, 16, list(candidate_pool), deadline
        )
        
        if result_points is not None and result_score > best_score:
            best_score = result_score
            best_points = result_points
    
    # Strategy 2: Simulated annealing from best found or 4x4 grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _evaluate(best_points)[1]
    
    # Refine with SA
    refined_points, refined_score = _simulated_annealing(best_points, deadline)
    
    if refined_score > best_score:
        best_points = refined_points
        best_score = refined_score
    
    return best_points
# EVOLVE-BLOCK-END
