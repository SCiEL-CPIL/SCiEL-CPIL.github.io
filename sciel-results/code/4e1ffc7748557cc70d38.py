"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 295.0  # safety margin before 300s limit


def _dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen_set):
    """Check if point p can be added to the set without creating an isosceles triangle."""
    for a in chosen_set:
        for b in chosen_set:
            if a == b:
                continue
            if _is_isosceles(p, a, b):
                return False
    return True


def _is_maximal(chosen_list):
    """Check if the set is maximal (no further point can be added)."""
    chosen_set = set(chosen_list)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set and _can_add(p, chosen_set):
            return False
    return True


def _greedy_randomized(seed_rng):
    """Generate a randomized greedy maximal set."""
    points = list(itertools.product(range(N), repeat=2))
    seed_rng.shuffle(points)
    
    chosen = []
    chosen_set = set()
    for p in points:
        if _can_add(p, chosen_set):
            chosen.append(p)
            chosen_set.add(p)
    
    # Ensure maximality by trying to add any remaining points
    for p in points:
        if p not in chosen_set:
            if _can_add(p, chosen_set):
                chosen.append(p)
                chosen_set.add(p)
    
    return chosen


def _try_remove(chosen_list, idx, chosen_set):
    """Try to remove point at idx and check if set remains maximal."""
    p = chosen_list[idx]
    new_set = chosen_set - {p}
    
    # Check if removing p allows any new point to be added
    # If so, the set is no longer maximal
    for q in itertools.product(range(N), repeat=2):
        if q not in new_set and _can_add(q, new_set):
            # Set would not be maximal after removal
            return None
    
    # If we get here, the set remains maximal without p
    new_list = chosen_list[:idx] + chosen_list[idx+1:]
    return new_list


def _local_search(chosen_list, max_iter=5000, rng=None):
    """Local search: try to remove points while maintaining maximality."""
    if rng is None:
        rng = random.Random()
    
    chosen_list = list(chosen_list)
    chosen_set = set(chosen_list)
    current_size = len(chosen_list)
    best_list = list(chosen_list)
    best_size = current_size
    
    for _ in range(max_iter):
        # Try removing a random point
        if not chosen_list:
            break
        idx = rng.randrange(len(chosen_list))
        result = _try_remove(chosen_list, idx, chosen_set)
        if result is not None:
            chosen_list = result
            chosen_set = set(chosen_list)
            current_size = len(chosen_list)
            if current_size < best_size:
                best_size = current_size
                best_list = list(chosen_list)
    
    return best_list


def construct():
    start_time = time.time()
    
    best_result = None
    best_size = float('inf')
    
    # Multiple randomized greedy starts with local search
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE:
        # Generate a randomized greedy maximal set
        greedy_set = _greedy_randomized(rng)
        
        # Apply local search to try to reduce size
        searched = _local_search(greedy_set, max_iter=2000, rng=rng)
        
        size = len(searched)
        if size < best_size:
            best_size = size
            best_result = searched
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    if best_result is None:
        # Fallback: simple greedy
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        best_result = chosen
    
    return best_result
# EVOLVE-BLOCK-END
