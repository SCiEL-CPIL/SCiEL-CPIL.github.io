"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _time_left(deadline, margin=2.0):
    return time.time() + margin < deadline

def _normalize_line(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        # vertical line x = c
        return (0, 0, p1[0])
    if dy == 0:
        # horizontal line y = c
        return (0, 1, p1[1])
    # general: dx*x + dy*y = c
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # ensure canonical sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * p1[0] + dy * p1[1]
    return (dx, dy, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _get_line_points(points, line_key):
    """Return the set of points lying on the given line."""
    if line_key is None:
        return set()
    dx, dy, c = line_key
    result = set()
    for i, p in enumerate(points):
        if dx * p[0] + dy * p[1] == c:
            result.add(i)
    return result

def _count_four_lines(points):
    """Count lines containing exactly 4 of the points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = {i, j}
            else:
                line_map[key].add(i)
                line_map[key].add(j)
    count = 0
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
        elif len(pts) >= 5:
            return -1  # invalid: five or more collinear
    return count

def _evaluate(points):
    """Evaluate the configuration. Returns (score, valid)."""
    n = len(points)
    # Check distinctness
    if len(set(tuple(p) for p in points)) != n:
        return (0, False)
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return (0, False)
    # Count four-point lines
    count = _count_four_lines(points)
    if count < 0:
        return (0, False)
    return (count, True)

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _neighborhood_move(points, idx):
    """Generate a random neighbor by moving one point."""
    new_points = [p[:] for p in points]
    # Small perturbation
    delta_x = random.randint(-20, 20)
    delta_y = random.randint(-20, 20)
    new_points[idx][0] = max(-1000, min(1000, points[idx][0] + delta_x))
    new_points[idx][1] = max(-1000, min(1000, points[idx][1] + delta_y))
    return new_points

def _large_move(points, idx):
    """Generate a random neighbor by relocating one point."""
    new_points = [p[:] for p in points]
    new_points[idx] = _random_point()
    return new_points

def construct():
    deadline = time.time() + 140.0  # safety margin
    best_score = 0
    best_points = None
    
    # Initial configuration: 4x4 grid
    initial = [[x, y] for x in range(4) for y in range(4)]
    score, valid = _evaluate(initial)
    if valid and score > best_score:
        best_score = score
        best_points = [p[:] for p in initial]
    
    current_points = [p[:] for p in initial]
    current_score = score
    
    # Simulated annealing
    temperature = 100.0
    cooling = 0.999
    
    while _time_left(deadline):
        # Choose move type
        if random.random() < 0.3:
            idx = random.randint(0, 15)
            candidate = _large_move(current_points, idx)
        else:
            idx = random.randint(0, 15)
            candidate = _neighborhood_move(current_points, idx)
        
        cand_score, cand_valid = _evaluate(candidate)
        
        if cand_valid:
            if cand_score >= current_score:
                # Accept
                current_points = candidate
                current_score = cand_score
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = [p[:] for p in candidate]
            else:
                # Probabilistic accept
                delta = cand_score - current_score
                if temperature > 0.01:
                    prob = min(1.0, 1.0 / (1.0 + abs(delta) / max(temperature, 0.01)))
                    if random.random() < prob:
                        current_points = candidate
                        current_score = cand_score
        else:
            # Invalid: penalize
            if random.random() < 0.05:
                current_points = candidate
                current_score = -10  # penalty
        
        temperature *= cooling
        
        # Occasionally restart from best
        if random.random() < 0.01 and best_points is not None:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            temperature = 50.0
    
    # Return best found
    if best_points is not None:
        return best_points
    return initial
# EVOLVE-BLOCK-END
