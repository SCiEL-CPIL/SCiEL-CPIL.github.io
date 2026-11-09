"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
from fractions import Fraction

# Precompute LCM of 1..22
_LCM = 232792560

def _verify_recip_dissociated(subset):
    """
    Return True if all 2^len(subset) reciprocal subset sums are distinct.
    Uses integer arithmetic with LCM as denominator.
    """
    if not subset:
        return True
    
    k = len(subset)
    values = [_LCM // i for i in subset]
    
    # Generate all subset sums using iterative doubling
    sums = {0}
    for v in values:
        new_sums = {s + v for s in sums}
        # If any new sum is already in sums, we have a collision
        # Actually, we need to check if the total count is 2^k
        # But we can early-exit if we detect a collision
        # However, a collision in the new_sums vs sums doesn't necessarily mean
        # a final collision. We need to check all sums.
        # Let's just build the full set and check size.
        sums = sums | new_sums
    
    return len(sums) == (1 << k)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    candidates = list(range(1, 23))
    
    # Start with a known good set (size 16 from current program's typical result)
    # But we don't know exactly which set. Let's find one quickly.
    
    # Strategy 1: Greedy with multiple orderings
    best_set = []
    best_size = 0
    
    # Try different orderings
    orderings = []
    
    # Order by i ascending
    orderings.append(sorted(candidates))
    # Order by i descending
    orderings.append(sorted(candidates, reverse=True))
    # Order by weight descending (small i first)
    orderings.append(sorted(candidates, key=lambda x: -(_LCM // x)))
    # Order by weight ascending (large i first)
    orderings.append(sorted(candidates, key=lambda x: _LCM // x))
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(50):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        current_set = []
        for i in ordering:
            if len(current_set) >= best_size:
                new_set = current_set + [i]
                if _verify_recip_dissociated(new_set):
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
            else:
                new_set = current_set + [i]
                if _verify_recip_dissociated(new_set):
                    current_set = new_set
    
    # Local search: try to add any missing element to best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in candidates:
            if i not in best_set:
                new_set = best_set + [i]
                if _verify_recip_dissociated(new_set):
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break
    
    # If best_set is empty (shouldn't happen), return a default
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
