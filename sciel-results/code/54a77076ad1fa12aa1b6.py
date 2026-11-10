"""Minimize largest 3-collinear-free subset for 20 points (no four collinear)."""
import time
import random
from math import gcd

_DEADLINE = 138.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_no_four_collinear(points):
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_triple_conflicts(points):
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_points = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_points.append(k)
            m = len(line_points)
            if m >= 3:
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = frozenset([line_points[a], line_points[b], line_points[c]])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _max_independent_set_size_fast(triples, n):
    """Fast bitmask-based branch and bound for max subset with no three collinear."""
    if not triples:
        return n
    
    # For each point, bitmask of points that form a collinear triple with it (paired)
    # Actually, we need: for each point i, the set of (j,k) pairs such that {i,j,k} is a triple
    # Better: for each point i, precompute which other points are "conflicting"
    
    # For each point i, build a bitmask of all points j such that there exists k 
    # where {i,j,k} is a triple. But we need the full triple check.
    
    # Alternative fast approach: 
    # For each point i, precompute a list of bitmasks representing "if I include i, 
    # I cannot include both j and k for each triple {i,j,k}"
    # This is complex. Let's use a different fast B&B.
    
    # Fast approach: use the fact that we can precompute for each point i,
    # the set of points that are in any triple with i.
    # Then use a greedy upper bound.
    
    # Actually, let's use a cleaner fast B&B with bitmasks:
    # Represent the set of "available" points as a bitmask.
    # For each point i, precompute conflict_mask[i] = bitmask of points j 
    # such that there's a triple {i, j, k} for some k.
    # But this doesn't capture the triple constraint properly.
    
    # Let me use a different approach: precompute for each point i,
    # the list of (j, k) pairs where {i,j,k} is a triple.
    # Then in B&B, when considering adding point i, check if any (j,k) 
    # with both j and k already selected exists.
    
    index_to_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for p in t:
            index_to_triples[p].append(idx)
    
    triples_set = [set(t) for t in triples]
    
    best = [0]
    current = [False] * n
    
    degrees = [len(index_to_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Precompute for each position in order, the index
    order_idx = [0] * n
    for pos, idx in enumerate(order):
        order_idx[idx] = pos
    
    def backtrack(pos, count, selected_mask):
        if _elapsed() > _DEADLINE - 3:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound
        if count + (n - pos) <= best[0]:
            return
        
        idx = order[pos]
        
        # Try including idx
        can_include = True
        for t_idx in index_to_triples[idx]:
            t = triples_set[t_idx]
            others = [p for p in t if p != idx]
            if len(others) == 2:
                if current[others[0]] and current[others[1]]:
                    can_include = False
                    break
        
        if can_include:
            current[idx] = True
            backtrack(pos + 1, count + 1, selected_mask | (1 << idx))
            current[idx] = False
        
        # Try excluding idx
        backtrack(pos + 1, count, selected_mask)
    
    backtrack(0, 0, 0)
    return best[0]

def _score(points):
    triples = _build_triple_conflicts(points)
    return _max_independent_set_size_fast(triples, len(points))

def _two_parabola_config(shift=1):
    points = []
    x_range = list(range(-5, 5))
    for x in x_range:
        points.append([x, x * x])
    for x in x_range:
        points.append([x, x * x + shift])
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) != 20:
        return None
    for x, y in unique:
        if abs(x) > 100 or abs(y) > 100:
            return None
    return unique

def _grid_config(m, n, ox=0, oy=0):
    """Generate m*n points on a grid, possibly with perturbations."""
    points = []
    seen = set()
    for i in range(m):
        for j in range(n):
            x = ox + i
            y = oy + j
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
    if len(points) != 20:
        return None
    return points

def _modular_config():
    """Points based on modular arithmetic to create many collinear triples."""
    # Try a 5x4 grid with some shifts
    points = []
    # Use a pattern that creates many 3-collinear sets
    # e.g., points on lines y = x, y = x+1, y = 2x, etc.
    configs = []
    
    # Configuration 1: 4 rows of 5 points each, shifted
    for shift in range(0, 10):
        pts = []
        for row in range(4):
            for col in range(5):
                x = col
                y = row * 3 + (shift if row % 2 == 0 else 0)
                pts.append([x, y])
        if len(set(tuple(p) for p in pts)) == 20:
            configs.append(pts)
    
    # Configuration 2: Based on finite field-like structure
    for a in range(-5, 6):
        for b in range(-5, 6):
            pts = []
            for i in range(5):
                for j in range(4):
                    x = i + a
                    y = j * 2 + (i if j % 2 == 0 else 0) + b
                    pts.append([x, y])
            if len(set(tuple(p) for p in pts)) == 20:
                all_in_bounds = all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in pts)
                if all_in_bounds:
                    configs.append(pts)
    
    return configs

def _random_valid_config():
    """Generate a random configuration and check validity."""
    for _ in range(100):
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points
    return None

def _perturb(points, radius=3):
    new_points = [list(p) for p in points]
    num_perturb = random.choice([1, 1, 2])
    for _ in range(num_perturb):
        idx = random.randint(0, 19)
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def _swap_points(points):
    """Swap two points with random perturbations."""
    new_points = [list(p) for p in points]
    i, j = random.sample(range(20), 2)
    dx1 = random.randint(-3, 3)
    dy1 = random.randint(-3, 3)
    dx2 = random.randint(-3, 3)
    dy2 = random.randint(-3, 3)
    nx1 = new_points[i][0] + dx1
    ny1 = new_points[i][1] + dy1
    nx2 = new_points[j][0] + dx2
    ny2 = new_points[j][1] + dy2
    if -100 <= nx1 <= 100 and -100 <= ny1 <= 100 and -100 <= nx2 <= 100 and -100 <= ny2 <= 100:
        new_points[i] = [nx1, ny1]
        new_points[j] = [nx2, ny2]
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def _local_search(points, score, time_budget):
    """Simulated annealing-like local search."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in points]
    best_score = score
    temperature = 2.0
    stall = 0
    
    while _elapsed() < time_budget:
        if random.random() < 0.3:
            candidate = _swap_points(current)
        else:
            candidate = _perturb(current)
        
        if candidate is None:
            stall += 1
            if stall > 100:
                break
            continue
        
        if not _check_no_four_collinear(candidate):
            stall += 1
            if stall > 100:
                break
            continue
        
        s = _score(candidate)
        
        # Accept if better, or with probability exp(-(s - current_score)/temperature)
        if s < current_score:
            current = candidate
            current_score = s
            if s < best_score:
                best = [list(p) for p in candidate]
                best_score = s
            stall = 0
            temperature = max(0.1, temperature * 0.95)
        elif s == current_score and random.random() < 0.3:
            current = candidate
            stall = 0
        else:
            # Accept worse with some probability (annealing)
            if random.random() < (0.05 / (1 + s - current_score)):
                current = candidate
                current_score = s
                temperature = max(0.1, temperature * 0.98)
            else:
                stall += 1
                temperature = max(0.1, temperature * 0.99)
        
        if stall > 200:
            break
    
    return best, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_score = 21
    
    # Phase 1: Evaluate structured initial configurations
    candidates = []
    
    # Two-parabola configs
    for shift in [1, 2, 3, -1, -2, 5, 7, 4, 6, -3, 8, -4, -5]:
        if _elapsed() > _DEADLINE - 30:
            break
        config = _two_parabola_config(shift)
        if config is None:
            continue
        if not _check_no_four_collinear(config):
            continue
        candidates.append(config)
    
    # Grid configs
    for m, n in [(4, 5), (5, 4)]:
        for ox in range(-2, 3):
            for oy in range(-2, 3):
                config = _grid_config(m, n, ox, oy)
                if config is not None:
                    if _check_no_four_collinear(config):
                        candidates.append(config)
    
    # Modular configs
    mod_configs = _modular_config()
    for config in mod_configs[:20]:
        if _check_no_four_collinear(config):
            candidates.append(config)
    
    # Random configs
    for _ in range(5):
        if _elapsed() > _DEADLINE - 30:
            break
        config = _random_valid_config()
        if config is not None:
            candidates.append(config)
    
    # Score all candidates
    for config in candidates:
        if _elapsed() > _DEADLINE - 25:
            break
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    # Phase 2: Local search from best configuration
    if best_points is not None:
        time_budget = _DEADLINE - 5
        best_points, best_score = _local_search(best_points, best_score, time_budget)
    
    # Phase 3: If we have time, try more random restarts
    if _elapsed() < _DEADLINE - 20 and best_score > 12:
        for _ in range(10):
            if _elapsed() > _DEADLINE - 10:
                break
            config = _random_valid_config()
            if config is None:
                continue
            s = _score(config)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in config]
                # Do a quick local search
                if _elapsed() < _DEADLINE - 10:
                    best_points, best_score = _local_search(best_points, best_score, _DEADLINE - 5)
    
    if best_points is None:
        # Fallback
        best_points = []
        for i in range(10):
            best_points.append([i, i * i])
            best_points.append([i, i * i + 1])
    
    return best_points
