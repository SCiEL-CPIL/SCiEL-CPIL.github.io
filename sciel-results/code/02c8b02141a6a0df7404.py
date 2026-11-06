"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

_start_time = time.time()
_DEADLINE = 320  # Safety margin before 330s kill

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
    for p in itertools.product(range(N), repeat=2):
        if p in chosen:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _is_minimal(chosen):
    """Check if no point can be removed while keeping the set maximal and no-isosceles."""
    chosen_set = set(chosen)
    for p in chosen:
        remaining = [q for q in chosen if q != p]
        # Check no-isosceles (should hold if original did)
        # Check maximality: can we add any point?
        for candidate in itertools.product(range(N), repeat=2):
            if candidate in chosen_set:
                continue
            if _can_add(candidate, remaining):
                return False  # p is not essential
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
    # Order points by distance from corners
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


def _local_search_minimize(chosen, max_iter=500):
    """Try to remove points while maintaining maximality and no-isosceles property."""
    chosen = list(chosen)
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for i in range(len(chosen)):
            if _time_exceeded():
                break
            # Try removing point i
            candidate = chosen[:i] + chosen[i+1:]
            # Check no-isosceles (should hold)
            # Check maximality
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen


def _local_search_add_remove(chosen, max_iter=200):
    """Try swap operations: remove one point, add another if possible."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for i in range(len(chosen)):
            if _time_exceeded():
                break
            # Try removing point i
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
    
    # Strategy 4: Multiple random greedy with different seeds
    for seed in range(20):
        if _time_exceeded():
            break
        chosen = _greedy_random(seed)
        chosen = _local_search_minimize(chosen)
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 5: More random seeds
    for seed in range(20, 80):
        if _time_exceeded():
            break
        chosen = _greedy_random(seed)
        chosen = _local_search_minimize(chosen)
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
    
    if _time_exceeded():
        return best
    
    # Strategy 6: Additional random seeds if time permits
    for seed in range(80, 200):
        if _time_exceeded():
            break
        chosen = _greedy_random(seed)
        chosen = _local_search_minimize(chosen)
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
    
    return best
# EVOLVE-BLOCK-END
