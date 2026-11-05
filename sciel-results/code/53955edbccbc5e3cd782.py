import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result

def _can_form_target(target, values):
    """
    Check if target can be formed as a non-empty subset sum of values (all positive).
    Uses meet-in-the-middle for efficiency.
    """
    n = len(values)
    if n == 0:
        return False
    
    # Sort values to help with pruning
    values = sorted(values)
    total = sum(values)
    if total < target:
        return False
    
    half = n // 2
    left = values[:half]
    right = values[half:]
    
    # Generate all subset sums for left half
    left_sums = set()
    left_sums.add(0)
    for mask in range(1, 1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        if s <= target:
            left_sums.add(s)
    
    # Generate all subset sums for right half and check
    right_sums = set()
    right_sums.add(0)
    for mask in range(1, 1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s <= target:
            right_sums.add(s)
    
    # Check if target can be formed from right alone
    if target in right_sums:
        return True
    
    # Check if target can be formed from left alone
    if target in left_sums:
        return True
    
    # Check if target can be formed from left + right
    for rs in right_sums:
        if rs > target:
            continue
        need = target - rs
        if need in left_sums:
            return True
    
    return False

def _is_valid_set(S):
    """
    Check if set S is valid: no reciprocal of a selected element equals the sum
    of reciprocals of a subset of the OTHER selected elements.
    """
    S = sorted(S)
    n = len(S)
    
    # Compute LCM of all elements
    lcm_val = _lcm_list(S)
    
    for idx, x in enumerate(S):
        # Target: 1/x scaled by LCM = lcm_val // x
        target_int = lcm_val // x
        # Values: 1/y scaled by LCM = lcm_val // y for y != x
        values_int = [lcm_val // y for i, y in enumerate(S) if i != idx]
        
        if _can_form_target(target_int, values_int):
            return False
    
    return True

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    # Start with the known good set {13, 14, ..., 24} of size 12
    best = set(range(13, 25))
    
    # Local search with multiple strategies
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Strategy 1: Try to add an element (size + 1)
        for c in range(1, 25):
            if time.time() > deadline:
                break
            if c in best:
                continue
            trial = best | {c}
            if _is_valid_set(trial):
                best = trial
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 2: Try swap (remove one, add one) to maintain size
        swapped = False
        for d in sorted(best):
            if time.time() > deadline:
                break
            for c in range(1, 25):
                if time.time() > deadline:
                    break
                if c in best:
                    continue
                trial = best - {d} | {c}
                if _is_valid_set(trial):
                    # Prefer swaps that increase sum (larger elements tend to be safer)
                    if sum(trial) > sum(best):
                        best = trial
                        improved = True
                        swapped = True
                        break
            if swapped:
                break
        
        if improved:
            continue
        
        # Strategy 3: Try double-swap (remove two, add two) for more exploration
        best_list = sorted(best)
        for i in range(len(best_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_list)):
                if time.time() > deadline:
                    break
                d1, d2 = best_list[i], best_list[j]
                for c1 in range(1, 25):
                    if time.time() > deadline:
                        break
                    if c1 in best:
                        continue
                    for c2 in range(c1 + 1, 25):
                        if time.time() > deadline:
                            break
                        if c2 in best:
                            continue
                        trial = best - {d1, d2} | {c1, c2}
                        if _is_valid_set(trial):
                            if sum(trial) > sum(best):
                                best = trial
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return sorted(best)
