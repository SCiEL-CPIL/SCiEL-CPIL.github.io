import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _can_form_target(target_int, candidate_vals):
    """Check if target_int can be formed as a non-empty subset sum of candidate_vals.
    Uses meet-in-the-middle with early termination."""
    n = len(candidate_vals)
    if n == 0:
        return False
    if target_int <= 0:
        return False
    if sum(candidate_vals) < target_int:
        return False
    
    # Quick check: single element
    if target_int in candidate_vals:
        return True
    
    # Meet in the middle for n <= 30
    if n <= 30:
        half = n // 2
        left = candidate_vals[:half]
        right = candidate_vals[half:]
        
        left_sums = set([0])
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        
        right_sums = set([0])
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}
        
        for ls in left_sums:
            if target_int - ls in right_sums:
                # Make sure it's a non-empty subset
                if ls == 0 and (target_int - ls) == 0:
                    continue
                return True
        return False
    else:
        # Fallback: check small subsets
        for i in range(n):
            for j in range(i+1, n):
                if candidate_vals[i] + candidate_vals[j] == target_int:
                    return True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if candidate_vals[i] + candidate_vals[j] + candidate_vals[k] == target_int:
                        return True
        return False

def _check_reciprocal_subsum(x, others, lcm_cache=None):
    """Check if 1/x equals sum of reciprocals of any non-empty subset of others.
    Returns True if violation exists."""
    if not others:
        return False
    
    all_vals = [x] + list(others)
    lcm_val = _lcm_list(all_vals)
    target_int = lcm_val // x
    candidate_vals = [lcm_val // y for y in others]
    
    return _can_form_target(target_int, candidate_vals)

def _is_valid_addition(x, current_set):
    """Check if adding x to current_set maintains the reciprocal-subsum-free property."""
    if not current_set:
        return True
    
    others = list(current_set)
    
    # Check 1: Is 1/x equal to sum of reciprocals of any non-empty subset of others?
    if _check_reciprocal_subsum(x, others):
        return False
    
    # Check 2: For each y in current_set, is 1/y equal to sum of reciprocals of any non-empty subset of (current_set ∪ {x}) \ {y}?
    for y in current_set:
        new_others = [z for z in others if z != y] + [x]
        if _check_reciprocal_subsum(y, new_others):
            return False
    
    return True

def _is_valid_set(S):
    """Full validation of a set S."""
    S = list(S)
    for i, x in enumerate(S):
        others = [y for j, y in enumerate(S) if j != i]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _try_swap(current_set, x_in, x_out):
    """Try removing x_out and adding x_in. Returns new set if valid, else None."""
    new_set = list(current_set)
    new_set.remove(x_out)
    new_set.append(x_in)
    new_set = set(new_set)
    
    if len(new_set) != len(current_set):
        return None
    
    if _is_valid_set(new_set):
        return new_set
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    result = set()
    
    # Greedy from largest to smallest
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_valid_addition(x, result):
            result.add(x)
    
    if not result:
        result = {24}
    
    # Local search: try swaps to improve
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current = list(result)
            for x_out in current:
                if time.time() > deadline - 5.0:
                    break
                for x_in in range(1, 25):
                    if x_in in result or x_in == x_out:
                        continue
                    new_set = _try_swap(result, x_in, x_out)
                    if new_set is not None and len(new_set) > len(result):
                        result = new_set
                        improved = True
                        break
                if improved:
                    break
    
    # Try to add more elements if possible
    if time.time() < deadline - 3.0:
        for x in range(24, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if x not in result and _is_valid_addition(x, result):
                result.add(x)
    
    if not result:
        result = {24}
    
    return sorted(result)
