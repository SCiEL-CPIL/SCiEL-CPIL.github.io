"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1..16]
_ALL_4SUBSETS = [tuple(sorted(c)) for c in combinations(range(1, 17), 4)]
N_ALL = len(_ALL_4SUBSETS)  # 1820

# Bitmask representation for fast intersection checks
def _to_mask(t):
    m = 0
    for x in t:
        m |= (1 << (x - 1))
    return m

_MASKS = [_to_mask(t) for t in _ALL_4SUBSETS]

def _pairwise_equal(a, b, c):
    """Check if three sets have all three pairwise intersections equal.
    This includes the empty-core case (all pairwise intersections empty)."""
    ab = _MASKS[a] & _MASKS[b]
    ac = _MASKS[a] & _MASKS[c]
    bc = _MASKS[b] & _MASKS[c]
    return ab == ac == bc

def _is_valid_addition(family_set, new_idx):
    """Check if adding new_idx to family_set violates the sunflower-free condition."""
    fam = list(family_set)
    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            if _pairwise_equal(fam[i], fam[j], new_idx):
                return False
    return True

def _greedy_random(seed, time_limit):
    """Greedy construction with random ordering."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(N_ALL)
    family = set()
    deadline = time.time() + time_limit
    
    for idx in order:
        if time.time() > deadline:
            break
        if _is_valid_addition(family, idx):
            family.add(idx)
    return family

def _local_search(family, time_limit, seed=0):
    """Try to improve family by swaps: remove one, add another."""
    rng = np.random.default_rng(seed)
    deadline = time.time() + time_limit
    current = set(family)
    best = set(current)
    
    # Precompute which pairs in current family are "close" to form sunflowers
    # For each element to remove, find candidates that could be added
    for step in range(10000):
        if time.time() > deadline:
            break
        
        # Randomly pick an element to remove
        if len(current) < 2:
            break
        rem = rng.choice(list(current))
        temp = current - {rem}
        
        # Find a random candidate to add
        cand = rng.integers(0, N_ALL)
        if cand in temp:
            continue
        if _is_valid_addition(temp, cand):
            new_fam = temp | {cand}
            # Check if this is an improvement (same size, but we might prefer
            # configurations that allow more additions later - for now just accept)
            current = new_fam
            if len(current) > len(best):
                best = set(current)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    best_family = set()
    
    # Phase 1: Multiple greedy runs with different seeds
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 1:
            break
        # Use a fraction of remaining time for this greedy run
        greedy_time = min(remaining * 0.3, 15)
        if greedy_time < 0.5:
            break
        
        fam = _greedy_random(seed, greedy_time)
        if len(fam) > len(best_family):
            best_family = set(fam)
        
        seed += 1
    
    # Phase 2: Local search on the best family found
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 2:
            improved = _local_search(best_family, remaining * 0.8, seed=42)
            if len(improved) > len(best_family):
                best_family = improved
    
    # Phase 3: More greedy runs if time remains
    if time.time() < deadline:
        seed = 100
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining < 1:
                break
            greedy_time = min(remaining * 0.2, 10)
            if greedy_time < 0.5:
                break
            fam = _greedy_random(seed, greedy_time)
            if len(fam) > len(best_family):
                best_family = set(fam)
            seed += 1
    
    # Convert back to sorted lists of 4-element subsets
    result = [list(_ALL_4SUBSETS[i]) for i in best_family]
    result.sort()
    return result
# EVOLVE-BLOCK-END
