"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
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
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Ensure consistent sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line: dx*y - dy*x = c
    c = dx * y1 - dy * x1
    g2 = math.gcd(abs(dx), abs(dy))
    if g2 > 1:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _point_on_line(p, line):
    """Check if point p lies on the given line (a, b, c) meaning a*x + b*y = c."""
    a, b, c = line
    x, y = p
    return a * x + b * y == c

def _compute_line_map(points):
    """Compute a map from line -> set of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            line = _normalize_line(points[i], points[j])
            if line not in line_map:
                line_map[line] = set()
            line_map[line].add(i)
            line_map[line].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points from a precomputed line map."""
    count = 0
    for line, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _max_collinear_from_line_map(line_map):
    """Return max number of collinear points from a precomputed line map."""
    if not line_map:
        return 0
    return max(len(s) for s in line_map.values())

def _score(points):
    """Return the number of lines with exactly 4 points, or -1 if invalid."""
    if len(points) != 16:
        return -1
    seen = set()
    for p in points:
        if len(p) != 2:
            return -1
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return -1
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
    
    line_map = _compute_line_map(points)
    if _max_collinear_from_line_map(line_map) >= 5:
        return -1
    return _score_from_line_map(line_map)

def _score_fast(points, line_map):
    """Fast scoring assuming points are already validated for distinctness/bounds."""
    new_line_map = _compute_line_map(points)
    if _max_collinear_from_line_map(new_line_map) >= 5:
        return -1
    return _score_from_line_map(new_line_map)

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
        if p not in points:
            points.append(p)
        attempts += 1
    return points

def _csp_build_points(max_time=5.0, seed=None):
    """
    Constraint-satisfaction-based construction: build 16 points one at a time,
    pruning any placement that would create a line with 5+ collinear points.
    Uses backtracking with randomized ordering.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    best_points = None
    best_score = -1
    
    for attempt in range(50):
        if time.time() >= deadline:
            break
        
        points = []
        used_positions = set()
        
        # Start with a small seed structure
        if attempt % 3 == 0:
            # Grid-like start
            base = random.choice([3, 4])
            off = random.randint(-5, 5)
            for i in range(base):
                for j in range(base):
                    p = [i + off, j + off]
                    if (p[0], p[1]) not in used_positions:
                        points.append(p)
                        used_positions.add((p[0], p[1]))
        elif attempt % 3 == 1:
            # Linear start
            slope = random.choice([0, 1, -1, 2, -2])
            intercept = random.randint(-20, 20)
            for i in range(6):
                x = i + random.randint(-5, 5)
                y = slope * x + intercept + random.randint(-3, 3)
                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                    if (x, y) not in used_positions:
                        points.append([x, y])
                        used_positions.add((x, y))
        else:
            # Random start
            for _ in range(5):
                p = _random_point()
                tp = tuple(p)
                if tp not in used_positions:
                    points.append(p)
                    used_positions.add(tp)
        
        # Now try to add more points via backtracking
        def backtrack(points, used_positions, depth, max_depth=16):
            if time.time() >= deadline:
                return False, points, used_positions
            
            if len(points) == max_depth:
                return True, points, used_positions
            
            if depth > 300:
                return False, points, used_positions
            
            # Generate candidate positions
            candidates = []
            
            # Try positions near existing points to encourage collinearity
            for p in points:
                for dx in range(-15, 16):
                    for dy in range(-15, 16):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = p[0] + dx, p[1] + dy
                        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                            if (nx, ny) not in used_positions:
                                candidates.append([nx, ny])
            
            # Also try some random positions
            for _ in range(20):
                p = _random_point()
                if tuple(p) not in used_positions:
                    candidates.append(p)
            
            random.shuffle(candidates)
            
            for cand in candidates:
                # Check if adding this point creates 5+ collinear
                test_points = points + [cand]
                line_map = _compute_line_map(test_points)
                if _max_collinear_from_line_map(line_map) >= 5:
                    continue
                
                # Accept and recurse
                new_used = used_positions | {tuple(cand)}
                success, result_points, result_used = backtrack(
                    test_points, new_used, depth + 1
                )
                if success:
                    return True, result_points, result_used
            
            return False, points, used_positions
        
        success, result_points, _ = backtrack(points, used_positions, 0)
        
        if success and len(result_points) == 16:
            sc = _score(result_points)
            if sc > best_score:
                best_score = sc
                best_points = [list(p) for p in result_points]
    
    return best_points, best_score

def _directed_local_search(points, max_iters=3000, deadline=None):
    """
    Directed local search: preferentially move points to positions that
    complete 3-point lines into 4-point lines.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        
        # Find 3-point lines that don't include point idx
        line_map = _compute_line_map(current)
        three_point_lines = []
        for line, idx_set in line_map.items():
            if len(idx_set) == 3 and idx not in idx_set:
                three_point_lines.append(line)
        
        candidates = []
        
        # Try to complete 3-point lines
        for line in three_point_lines:
            a, b, c = line
            # Find integer points on this line within bounds
            # a*x + b*y = c
            # Try a range of x values
            x_min = max(-1000, -c // a if a != 0 else -1000)
            x_max = min(1000, -c // a if a != 0 else 1000)
            if a == 0:
                # b*y = c -> y = c/b
                if b != 0 and c % b == 0:
                    y = c // b
                    if -1000 <= y <= 1000:
                        for x in range(-1000, 1001, 10):
                            if (x, y) not in [tuple(p) for p in current if p != old_point]:
                                candidates.append([x, y])
            else:
                # For each x, check if y is integer
                step = max(1, 2000 // max(1, abs(a) * 20))
                for x in range(max(-1000, x_min), min(1000, x_max) + 1, step):
                    num = c - a * x
                    if b == 0:
                        continue
                    if num % b == 0:
                        y = num // b
                        if -1000 <= y <= 1000:
                            if (x, y) not in [tuple(p) for p in current if p != old_point]:
                                candidates.append([x, y])
        
        # Also add some random candidates
        for _ in range(5):
            p = _random_point()
            if tuple(p) not in [tuple(q) for q in current if q != old_point]:
                candidates.append(p)
        
        if not candidates:
            continue
        
        random.shuffle(candidates)
        
        best_candidate = None
        best_candidate_score = current_score
        
        for cand in candidates[:20]:  # Limit candidates per iteration
            current[idx] = cand
            new_score = _score_fast(current, line_map)
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = cand
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        
        if best_candidate is not None and best_candidate_score > current_score:
            current[idx] = best_candidate
            current_score = best_candidate_score
        else:
            current[idx] = old_point
    
    return best, best_score

def _local_search(points, max_iters=5000, deadline=None):
    """Perform local search to improve the configuration."""
    if deadline is None:
        deadline = time.time() + 10
    
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
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
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
    
    return best, best_score

def _simulated_annealing(points, start_temp=1000, end_temp=1, max_iters=10000, deadline=None):
    """Perform simulated annealing."""
    if deadline is None:
        deadline = time.time() + 10
    
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
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
                prob = math.exp(delta / temp)
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

def _exhaustive_local_polish(points, max_rounds=20, deadline=None):
    """Try all possible single-point moves to improve."""
    if deadline is None:
        deadline = time.time() + 10
    
    current = [list(p) for p in points]
    current_score = _score(current)
    if current_score < 0:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for round_num in range(max_rounds):
        if time.time() >= deadline:
            break
        
        improved = False
        for idx in range(16):
            old_point = list(current[idx])
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
    
    # Phase 1: CSP-based construction (constraint satisfaction with backtracking)
    csp_points, csp_score = _csp_build_points(max_time=10.0)
    if csp_points is not None and csp_score > best_score:
        best_score = csp_score
        best_points = [list(p) for p in csp_points]
    
    # Phase 2: Start with the 4x4 grid as a baseline
    grid_points = _generate_grid_config(4, 0)
    score = _score(grid_points)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in grid_points]
    
    # Phase 3: Directed local search from best configuration
    if time.time() < deadline:
        improved, imp_score = _directed_local_search(best_points, max_iters=5000, deadline=deadline)
        if imp_score > best_score:
            best_score = imp_score
            best_points = [list(p) for p in improved]
    
    # Phase 4: Multiple restarts with different strategies
    for restart in range(15):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            start_config = [list(p) for p in best_points]
        elif restart % 3 == 1:
            # CSP-based start
            csp_pts, _ = _csp_build_points(max_time=3.0, seed=restart * 1000 + 42)
            if csp_pts is not None and len(csp_pts) == 16:
                start_config = [list(p) for p in csp_pts]
            else:
                start_config = _generate_diverse_config()
        elif restart % 3 == 2:
            # Slightly perturbed grid
            base = random.choice([3, 4, 5])
            start_config = _generate_grid_config(base, random.randint(-2, 2))
            while len(start_config) < 16:
                p = _random_point()
                if p not in start_config:
                    start_config.append(p)
            start_config = start_config[:16]
        else:
            # Diverse random
            start_config = _generate_diverse_config()
        
        # Directed local search
        if time.time() < deadline:
            improved, imp_score = _directed_local_search(start_config, max_iters=3000, deadline=deadline)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
        
        # Standard local search
        if time.time() < deadline:
            improved, imp_score = _local_search(improved, max_iters=2000, deadline=deadline)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
        
        # Simulated annealing
        if time.time() < deadline:
            improved, imp_score = _simulated_annealing(improved, start_temp=500, end_temp=1, max_iters=3000, deadline=deadline)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
        
        # Final polish
        if time.time() < deadline:
            polished, pol_score = _exhaustive_local_polish(best_points, max_rounds=10, deadline=deadline)
            if pol_score > best_score:
                best_score = pol_score
                best_points = [list(p) for p in polished]
    
    # Final polish on the best configuration
    if time.time() < deadline:
        polished, pol_score = _exhaustive_local_polish(best_points, max_rounds=15, deadline=deadline)
        if pol_score > best_score:
            best_score = pol_score
            best_points = [list(p) for p in polished]
    
    # Ensure we have a valid result
    if best_points is None or not _validate(best_points):
        return _generate_grid_config(4, 0)
    
    return best_points

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
    
    line_map = _compute_line_map(points)
    if _max_collinear_from_line_map(line_map) >= 5:
        return False
    return True
# EVOLVE-BLOCK-END
