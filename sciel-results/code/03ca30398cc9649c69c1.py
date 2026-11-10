"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _can_form_target(target_int, candidate_vals):
    """Check if target_int can be formed as a subset sum of candidate_vals."""
    n = len(candidate_vals)
    if n == 0:
        return target_int == 0
    if n <= 22:
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
                return True
        return False
    else:
        # Fallback for larger sets
        for v in candidate_vals:
            if v == target_int:
                return True
        for i in range(n):
            for j in range(i + 1, n):
                if candidate_vals[i] + candidate_vals[j] == target_int:
                    return True
        return False

def _is_valid_set(S):
    """Check if no reciprocal in S equals sum of reciprocals of a subset of S without that element."""
    S = list(S)
    n = len(S)
    if n <= 1:
        return True
    for i, x in enumerate(S):
        others = S[:i] + S[i+1:]
        if not others:
            continue
        all_vals = [x] + others
        lcm_val = _lcm_list(all_vals)
        target_int = lcm_val // x
        candidate_vals = [lcm_val // y for y in others]
        if _can_form_target(target_int, candidate_vals):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    # Start with a reasonable baseline: elements 13-24
    current = list(range(13, 25))
    
    # Greedily try to add smaller elements
    for x in range(12, 0, -1):
        if time.time() > deadline:
            break
        test = current + [x]
        if _is_valid_set(test):
            current = test
    
    # Local search: try to improve by adding or swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(current)
        
        # Try adding any element not in current
        for x in range(1, 25):
            if x in in_set:
                continue
            if time.time() > deadline:
                break
            test = current + [x]
            if _is_valid_set(test):
                current = test
                in_set.add(x)
                improved = True
        
        # Try swapping to increase size
        for x in range(1, 25):
            if x in in_set:
                continue
            for y in list(current):
                if time.time() > deadline:
                    break
                new_set = [z for z in current if z != y] + [x]
                if len(new_set) > len(current) and _is_valid_set(new_set):
                    current = new_set
                    in_set = set(current)
                    improved = True
                    break
    
    if not current:
        current = [13]
    
    return current

# EVOLVE-BLOCK-END
