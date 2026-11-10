"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _normalize_line(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    g = __import__('math').gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Ensure consistent sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line: dx*y - dy*x = c
    c = dx * y1 - dy * x1
    g2 = __import__('math').gcd(abs(dx), abs(dy))
    if g2 > 1:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _count_4_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            line = _normalize_line(points[i], points[j])
            line_counts[line] = line_counts.get(line, 0) + 1
    
    count = 0
    for line, pair_count in line_counts.items():
        # Number of points on this line
        # If there are C(k,2) pairs, then k points
        # We need k = 4 exactly
        # Solve C(k,2) = pair_count
        # k*(k-1)/2 = pair_count
        # k^2 - k - 2*pair_count = 0
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        import math
        disc = 1 + 8 * pair_count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            if k == 4:
                count += 1
    return count

def _no_five_collinear(points):
    """Check that no five points are collinear."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            line = _normalize_line(points[i], points[j])
            line_counts[line] = line_counts.get(line, 0) + 1
    
    for line, pair_count in line_counts.items():
        import math
        disc = 1 + 8 * pair_count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            if k >= 5:
                return False
    return True

def _validate(points):
    """Validate that points are distinct, within bounds, and no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if not _no_five_collinear(points):
        return False
    return True

def _score(points):
    """Return the number of lines with exactly 4 points, or -1 if invalid."""
    if not _validate(points):
        return -1
    return _count_4_lines(points)

def _random_point():
    """Generate a random integer point within bounds."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _generate_grid_config(base=4, offset=0):
    """Generate a grid-like configuration."""
    points = []
    for i in range(base):
        for j in range(base):
            points.append([i + offset, j + offset])
    return points

def _generate_diverse_config():
    """Generate a diverse random configuration."""
    points = []
    attempts = 0
    while len(points) < 16 and attempts < 10000:
        p = _random_point()
        # Check distinctness
        if p not in points:
            points.append(p)
        attempts += 1
    return points

def _local_search(points, max_iters=5000):
    """Perform local search to improve the configuration."""
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        new_point = _random_point()
        
        if new_point == old_point:
            continue
        
        # Check if new point is distinct from others
        is_distinct = True
        for j, p in enumerate(current):
            if j != idx and p == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
        
        current[idx] = new_point
        new_score = _score(current)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
    
    return best, best_score

def _simulated_annealing(points, start_temp=1000, end_temp=1, max_iters=10000):
    """Perform simulated annealing."""
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        new_point = _random_point()
        
        if new_point == old_point:
            continue
        
        is_distinct = True
        for j, p in enumerate(current):
            if j != idx and p == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
        
        current[idx] = new_point
        new_score = _score(current)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current_score = new_score
                else:
                    current[idx] = old_point
            else:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
    
    return best, best_score

def _exhaustive_local_polish(points, max_rounds=20):
    """Try all possible single-point moves to improve."""
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for round_num in range(max_rounds):
        improved = False
        for idx in range(16):
            old_point = list(current[idx])
            # Try moving to nearby points
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_point = [old_point[0] + dx, old_point[1] + dy]
                    if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
                        continue
                    
                    is_distinct = True
                    for j, p in enumerate(current):
                        if j != idx and p == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    
                    current[idx] = new_point
                    new_score = _score(current)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best = [list(p) for p in current]
                        improved = True
                    else:
                        current[idx] = old_point
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1
    
    # Start with the 4x4 grid
    grid_points = _generate_grid_config(4, 0)
    score = _score(grid_points)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in grid_points]
    
    # Try local search from grid
    if time.time() < deadline:
        improved, imp_score = _local_search(grid_points, max_iters=3000)
        if imp_score > best_score:
            best_score = imp_score
            best_points = [list(p) for p in improved]
    
    # Try simulated annealing from diverse configurations
    for restart in range(10):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            start_config = [list(p) for p in best_points]
        elif restart % 2 == 1:
            start_config = _generate_diverse_config()
        else:
            # Slightly perturbed grid
            base = random.choice([3, 4, 5])
            start_config = _generate_grid_config(base, random.randint(-2, 2))
            # Ensure 16 distinct points
            while len(start_config) < 16:
                p = _random_point()
                if p not in start_config:
                    start_config.append(p)
            start_config = start_config[:16]
        
        # First do local search
        if time.time() < deadline:
            improved, imp_score = _local_search(start_config, max_iters=2000)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
        
        # Then simulated annealing
        if time.time() < deadline:
            improved, imp_score = _simulated_annealing(improved, start_temp=500, end_temp=1, max_iters=5000)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
        
        # Final polish
        if time.time() < deadline:
            polished, pol_score = _exhaustive_local_polish(best_points, max_rounds=10)
            if pol_score > best_score:
                best_score = pol_score
                best_points = [list(p) for p in polished]
    
    # Final polish on the best configuration
    if time.time() < deadline:
        polished, pol_score = _exhaustive_local_polish(best_points, max_rounds=15)
        if pol_score > best_score:
            best_score = pol_score
            best_points = [list(p) for p in polished]
    
    # Ensure we have a valid result
    if best_points is None or not _validate(best_points):
        # Fallback to grid
        return _generate_grid_config(4, 0)
    
    return best_points
# EVOLVE-BLOCK-END
