"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
import math

DEADLINE = 320  # seconds, safety margin before 330s kill


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _sq_dist(a, b)
    d2 = _sq_dist(a, c)
    d3 = _sq_dist(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_valid_set(chosen):
    """Check that no three points form an isosceles triangle."""
    for a, b, c in itertools.combinations(chosen, 3):
        if _is_iso(a, b, c):
            return False
    return True


def _can_add(chosen_set, p):
    """Check if point p can be added to chosen_set without creating isosceles."""
    for a, b in itertools.combinations(chosen_set, 2):
        if _is_iso(p, a, b):
            return False
    return True


def _is_maximal(chosen_set):
    """Check if no further grid point can be added."""
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p not in chosen_set:
                if _can_add(chosen_set, p):
                    return False
    return True


def _greedy_construct(order, start_set=None):
    """Greedy construction following given order, optionally starting from start_set."""
    if start_set is not None:
        chosen = list(start_set)
    else:
        chosen = []
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(chosen_set, p):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _shrink(chosen):
    """Remove any point whose removal keeps maximality and validity. Returns smaller set."""
    chosen_set = set(chosen)
    improved = True
    while improved:
        improved = False
        for p in list(chosen_set):
            candidate = chosen_set - {p}
            if _is_valid_set(candidate):
                if _is_maximal(candidate):
                    chosen_set = candidate
                    improved = True
                    break
    return list(chosen_set)


def _random_order(seed=None):
    rng = random.Random(seed)
    points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(points)
    return points


def _perturb(chosen, rng):
    """Randomly remove 1-3 points and try to repair by adding others."""
    chosen_set = set(chosen)
    k = rng.randint(1, 3)
    to_remove = rng.sample(list(chosen_set), min(k, len(chosen_set)))
    for p in to_remove:
        chosen_set.discard(p)
    
    # Try to add back points in random order to maximize size reduction
    all_points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(all_points)
    for p in all_points:
        if p not in chosen_set:
            if _can_add(chosen_set, p):
                chosen_set.add(p)
    return list(chosen_set)


def construct():
    start_time = time.time()
    best = None
    best_size = float('inf')
    
    # Initial greedy in row-major order
    row_major = list(itertools.product(range(N), repeat=2))
    initial = _greedy_construct(row_major)
    initial = _shrink(initial)
    best = initial
    best_size = len(initial)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best
    
    # Multiple random restarts
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        seed = 42 + attempts
        order = _random_order(seed)
        
        # Greedy with random order
        candidate = _greedy_construct(order)
        candidate = _shrink(candidate)
        
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
        
        # Try local search: perturb and repair
        if len(best) > 0:
            rng = random.Random(seed)
            perturbed = _perturb(best, rng)
            perturbed = _shrink(perturbed)
            if len(perturbed) < best_size:
                best = perturbed
                best_size = len(perturbed)
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 30:
            break
    
    return best
# EVOLVE-BLOCK-END
