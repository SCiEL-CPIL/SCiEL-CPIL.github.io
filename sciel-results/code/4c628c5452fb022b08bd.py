"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _can_add_fast(p, chosen):
    """
    Check if p can be added to chosen without forming an isosceles triangle.
    Optimized to check O(k) for 'apex at p' and O(k^2) for other cases with early exit.
    """
    if not chosen:
        return True
    
    dists = []
    # Check if d(p,a) == d(p,b) for any a,b in chosen
    for a in chosen:
        d = _dist2(p, a)
        if d in dists:
            return False
        dists.append(d)
        
    # Check if d(p,a) == d(a,b) or d(p,b) == d(a,b) for any pair a,b in chosen
    k = len(chosen)
    for i in range(k):
        a = chosen[i]
        dp_a = dists[i]
        for j in range(i + 1, k):
            b = chosen[j]
            d_ab = _dist2(a, b)
            if dp_a == d_ab or dists[j] == d_ab:
                return False
    return True


def _greedy_maximal(order, chosen, deadline):
    """Greedy maximal construction given an ordering of points."""
    for p in order:
        if time.time() > deadline:
            return None
        if _can_add_fast(p, chosen):
            chosen.append(p)
    return chosen


def _is_maximal(chosen, all_points):
    """Check if no point can be added to chosen without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add_fast(p, chosen):
            return False
    return True


def _prune(chosen, all_points, deadline):
    """Try to remove points that are not essential for maximality."""
    if time.time() > deadline:
        return chosen
    chosen = list(chosen)
    i = 0
    while i < len(chosen):
        if time.time() > deadline:
            break
        p = chosen[i]
        candidate = chosen[:i] + chosen[i+1:]
        if _is_maximal(candidate, all_points):
            chosen = candidate
            # Don't increment i since we removed an element
        else:
            i += 1
    return chosen


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    result = _greedy_maximal(all_points, [], deadline)
    if result is not None:
        result = _prune(result, all_points, deadline)
        best = result
        best_size = len(result)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    num_trials = 5000
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        # Shuffle the order
        order = all_points[:]
        rng.shuffle(order)
        chosen = []
        result = _greedy_maximal(order, chosen, deadline)
        if result is None:
            break
        # Prune the result
        result = _prune(result, all_points, deadline)
        if len(result) < best_size:
            best = result
            best_size = len(result)
            if best_size <= 8:
                break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
