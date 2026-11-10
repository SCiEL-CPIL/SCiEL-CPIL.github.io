"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: 90s wall-clock, return best with safety margin (85s)
_DEADLINE = 85.0

def _check_validity(selected):
    """Check if no element divides the sum of any 3 distinct other selected elements."""
    n = len(selected)
    if n < 4:
        return True
    # Precompute sums of all triples
    # For efficiency, we can just check directly
    for i, a in enumerate(selected):
        # Check if a divides sum of any 3 distinct others
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        # Iterate over all triples of others
        for t in combinations(others, 3):
            s = t[0] + t[1] + t[2]
            if s % a == 0:
                return False
    return True

def _check_validity_fast(selected):
    """Faster validity check using precomputed triple sums or early termination."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        # Early termination: if any triple sum is divisible by a
        for t in combinations(others, 3):
            if (t[0] + t[1] + t[2]) % a == 0:
                return False
    return True

def _greedy_construct(start_order=None, deadline=85.0):
    """Greedy construction: add elements in given order if valid."""
    if start_order is None:
        start_order = list(range(192, 0, -1))
    selected = []
    start_time = time.time()
    for x in start_order:
        if time.time() > deadline:
            break
        candidate = selected + [x]
        if _check_validity(candidate):
            selected.append(x)
    return selected

def _local_search(selected, deadline=85.0):
    """Try to improve the set via local search: remove and re-add elements."""
    start_time = time.time()
    best = list(selected)
    best_size = len(best)
    
    # Try removing an element and adding a different one
    # Or try swapping
    n = len(best)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            # Try swapping best[i] and best[j] with a new element
            # This is complex, so let's just try adding elements not in the set
            pass
    
    # Try adding elements not in the set
    current_set = set(best)
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        candidate = best + [x]
        if _check_validity(candidate):
            best.append(x)
            current_set.add(x)
    
    return best

def construct():
    start_time = time.time()
    
    # First, try the greedy approach from high to low
    best = _greedy_construct(deadline=_DEADLINE)
    
    # Try different random orderings to see if we can get a better set
    random.seed(42)
    for _ in range(10):
        if time.time() > _DEADLINE:
            break
        order = list(range(192, 0, -1))
        random.shuffle(order)
        candidate = _greedy_construct(start_order=order, deadline=_DEADLINE)
        if len(candidate) > len(best):
            best = candidate
    
    # Try local search on the best set
    if time.time() < _DEADLINE:
        improved = _local_search(best, deadline=_DEADLINE)
        if len(improved) > len(best):
            best = improved
    
    # Ensure we return a valid set
    if not _check_validity(best):
        # Fallback: rebuild greedily
        best = _greedy_construct(deadline=_DEADLINE)
    
    return best
# EVOLVE-BLOCK-END
