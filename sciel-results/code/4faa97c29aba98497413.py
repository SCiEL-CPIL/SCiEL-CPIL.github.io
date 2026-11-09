"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _subset_sum_exists(elements, target):
    """Check if any nonempty subset of elements sums to target (exact)."""
    n = len(elements)
    if n == 0:
        return False
    if target == Fraction(0, 1):
        return False
    total = sum(elements)
    if target > total:
        return False
    if target < min(elements):
        return False
    achievable = {Fraction(0, 1)}
    for e in elements:
        new_sums = set()
        for s in achievable:
            new_s = s + e
            new_sums.add(new_s)
        achievable |= new_sums
        if target in achievable:
            return True
    return False

def _is_valid_set(S):
    """Check if set S is reciprocal-subsum-free."""
    S = list(S)
    n = len(S)
    if n == 0:
        return True
    recip = [Fraction(1, x) for x in S]
    for i in range(n):
        target = recip[i]
        others = recip[:i] + recip[i+1:]
        if _subset_sum_exists(others, target):
            return False
    return True

def _can_add(current_set, new_elem):
    """Check if adding new_elem to current_set maintains validity."""
    S = list(current_set)
    if not S:
        return True
    recip_new = Fraction(1, new_elem)
    recip_old = [Fraction(1, x) for x in S]
    
    if _subset_sum_exists(recip_old, recip_new):
        return False
    
    for i, old_elem in enumerate(S):
        target_diff = recip_old[i] - recip_new
        if target_diff <= 0:
            continue
        others_without_i = recip_old[:i] + recip_old[i+1:]
        if _subset_sum_exists(others_without_i, target_diff):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    universe = list(range(1, 25))
    best_set = set()
    
    orderings = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [x for x in range(1, 25) if x % 2 == 1] + [x for x in range(1, 25) if x % 2 == 0],
        [x for x in range(1, 25) if x > 12] + [x for x in range(1, 25) if x <= 12],
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        current = set()
        for elem in ordering:
            if time.time() > deadline:
                break
            if _can_add(current, elem):
                current.add(elem)
        if len(current) > len(best_set):
            best_set = set(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in universe:
            if time.time() > deadline:
                break
            if elem not in best_set:
                if _can_add(best_set, elem):
                    best_set.add(elem)
                    improved = True
    
    if time.time() < deadline:
        for a in list(best_set):
            if time.time() > deadline:
                break
            trial = set(best_set) - {a}
            for b in universe:
                if time.time() > deadline:
                    break
                if b in best_set:
                    continue
                if _can_add(trial, b):
                    trial.add(b)
                    for c in universe:
                        if time.time() > deadline:
                            break
                        if c in trial:
                            continue
                        if _can_add(trial, c):
                            trial.add(c)
                    if len(trial) > len(best_set):
                        best_set = set(trial)
    
    # Final validation
    if not _is_valid_set(best_set):
        # Fallback to seed
        return list(range(13, 25))
    
    return list(best_set)
# EVOLVE-BLOCK-END
