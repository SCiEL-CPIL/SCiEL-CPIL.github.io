"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

_start_time = time.time()
_DEADLINE = 315  # Safety margin before 330s kill


def _time_exceeded():
    return time.time() - _start_time > _DEADLINE


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(point, chosen):
    """Check if point can be added to chosen without creating isosceles triangles."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _isosceles(point, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen):
    """Check if no grid point can be added."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _greedy_random(seed=None):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(points)
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_corner_first():
    """Greedy construction starting from corners outward."""
    corners = [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1)]
    points = list(itertools.product(range(N), repeat=2))
    points.sort(key=lambda p: min(_dist2(p, c) for c in corners))
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_center_first():
    """Greedy construction starting from center outward."""
    center = (N // 2, N // 2)
    points = list(itertools.product(range(N), repeat=2))
    points.sort(key=lambda p: _dist2(p, center))
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_diagonal():
    """Greedy construction in diagonal order."""
    points = list(itertools.product(range(N), repeat=2))
    points.sort(key=lambda p: (p[0] + p[1], p[0]))
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_spiral():
    """Greedy construction in spiral order from center."""
    center = (N // 2, N // 2)
    points = list(itertools.product(range(N), repeat=2))
    # Sort by distance from center, then by angle
    points.sort(key=lambda p: (_dist2(p, center), (p[0] - center[0]) * (p[1] - center[1])))
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _local_search_minimize(chosen, max_iter=2000):
    """Try to remove points while maintaining maximality and no-isosceles property.
    Optimized: precompute which points are blocked by the current set."""
    chosen = list(chosen)
    improved = True
    iter_count = 0
    while improved and not _time_exceeded() and iter_count < max_iter:
        improved = False
        iter_count += 1
        chosen_set = set(chosen)
        for i in range(len(chosen)):
            if _time_exceeded():
                break
            # Try removing point i
            candidate = chosen[:i] + chosen[i+1:]
            candidate_set = set(candidate)
            # Quick check: is the set still maximal?
            # For each unchosen point, check if it can be added
            is_max = True
            for p in itertools.product(range(N), repeat=2):
                if p in candidate_set:
                    continue
                if _can_add(p, candidate):
                    is_max = False
                    break
            if is_max:
                chosen = candidate
                improved = True
                break
    return chosen


def _local_search_aggressive(chosen, max_iter=500):
    """More aggressive local search: try multiple removals per pass."""
    chosen = list(chosen)
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for i in range(len(chosen)):
            if _time_exceeded():
                break
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen


def construct():
    global _start_time
    _start_time = time.time()
    
    best = None
    best_size = float('inf')
    
    # Strategy 1: Original row-major greedy
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if _can_add(p, chosen):
            chosen.append(p)
    chosen = _local_search_minimize(chosen)
    if len(chosen) < best_size:
        best = chosen
        best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 2: Corner-first greedy
    chosen = _greedy_corner_first()
    chosen = _local_search_minimize(chosen)
    if len(chosen) < best_size:
        best = chosen
        best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 3: Center-first greedy
    chosen = _greedy_center_first()
    chosen = _local_search_minimize(chosen)
    if len(chosen) < best_size:
        best = chosen
        best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 4: Diagonal greedy
    chosen = _greedy_diagonal()
    chosen = _local_search_minimize(chosen)
    if len(chosen) < best_size:
        best = chosen
        best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 5: Spiral greedy
    chosen = _greedy_spiral()
    chosen = _local_search_minimize(chosen)
    if len(chosen) < best_size:
        best = chosen
        best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 6: Multiple random greedy with different seeds
    for seed in range(100):
        if _time_exceeded():
            break
        chosen = _greedy_random(seed)
        chosen = _local_search_minimize(chosen)
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 7: More random seeds with aggressive local search
    for seed in range(100, 250):
        if _time_exceeded():
            break
        chosen = _greedy_random(seed)
        chosen = _local_search_aggressive(chosen)
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 8: Final refinement on best solution
    if best is not None and not _time_exceeded():
        best = _local_search_minimize(best)
        best = _local_search_aggressive(best)
    
    return best
# EVOLVE-BLOCK-END
