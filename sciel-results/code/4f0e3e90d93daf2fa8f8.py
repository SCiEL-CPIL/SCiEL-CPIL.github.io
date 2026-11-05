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
    Uses iterative set-based generation for efficiency.
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
    
    # Generate all subset sums for left half using iterative method
    left_sums = {0}
    for v in left:
        new_sums = set()
        for s in left_sums:
            ns = s + v
            if ns <= target:
                new_sums.add(ns)
        left_sums |= new_sums
    
    # Generate all subset sums for right half
    right_sums = {0}
    for v in right:
        new_sums = set()
        for s in right_sums:
            ns = s + v
            if ns <= target:
                new_sums.add(ns)
        right_sums |= new_sums
    
    # Check if target can be formed
    if target in right_sums or target in left_sums:
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
        # Try all candidates, keep the best (largest element that works)
        candidates = []
        for c in range(1, 25):
            if time.time() > deadline:
                break
            if c in best:
                continue
            trial = best | {c}
            if _is_valid_set(trial):
                candidates.append(c)
        
        if candidates:
            # Prefer larger elements (they're generally "safer")
            best_c = max(candidates)
            best = best | {best_c}
            improved = True
            continue
        
        # Strategy 2: Try swap (remove one, add one) to maintain size
        swapped = False
        best_swap = None
        best_swap_sum = sum(best)
        for d in sorted(best, reverse=True):  # Try removing larger elements first
            if time.time() > deadline:
                break
            for c in range(24, 0, -1):  # Try adding larger elements first
                if time.time() > deadline:
                    break
                if c in best:
                    continue
                trial = best - {d} | {c}
                if _is_valid_set(trial):
                    s = sum(trial)
                    if s > best_swap_sum:
                        best_swap = trial
                        best_swap_sum = s
                        swapped = True
                        break
            if swapped:
                break
        
        if swapped and best_swap:
            best = best_swap
            improved = True
            continue
        
        # Strategy 3: Try double-swap (remove two, add two) for more exploration
        best_list = sorted(best, reverse=True)
        best_double_swap = None
        best_double_sum = sum(best)
        found = False
        for i in range(len(best_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_list)):
                if time.time() > deadline:
                    break
                d1, d2 = best_list[i], best_list[j]
                for c1 in range(24, 0, -1):
                    if time.time() > deadline:
                        break
                    if c1 in best:
                        continue
                    for c2 in range(c1 - 1, 0, -1):
                        if time.time() > deadline:
                            break
                        if c2 in best:
                            continue
                        trial = best - {d1, d2} | {c1, c2}
                        if _is_valid_set(trial):
                            s = sum(trial)
                            if s > best_double_sum:
                                best_double_swap = trial
                                best_double_sum = s
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if found and best_double_swap:
            best = best_double_swap
            improved = True
            continue
    
    return sorted(best)
