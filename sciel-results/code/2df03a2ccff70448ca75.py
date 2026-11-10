"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = lst[0]
    for x in lst[1:]:
        result = _lcm(result, x)
    return result

def _is_subset_sum(target, values):
    """Check if target is a subset sum of values. Returns True if yes."""
    if target == 0:
        return True
    if not values:
        return False
    if target < 0:
        return False
    
    n = len(values)
    if n <= 20:
        possible = {0}
        for v in values:
            new_possible = set()
            for s in possible:
                new_possible.add(s + v)
            possible = possible | new_possible
        return target in possible
    else:
        # Meet in the middle
        half = n // 2
        left = values[:half]
        right = values[half:]
        
        left_sums = {0}
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums = left_sums | new_sums
        
        right_sums = {0}
        for v in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + v)
            right_sums = right_sums | new_sums
        
        for ls in left_sums:
            if (target - ls) in right_sums:
                return True
        return False

def _is_valid_set(S):
    """Check if set S satisfies the reciprocal-subsum-free property."""
    S_list = list(S)
    for x in S_list:
        others = [y for y in S_list if y != x]
        if not others:
            continue
        all_vals = [x] + others
        lcm_val = _lcm_list(all_vals)
        target_int = lcm_val // x
        candidate_vals = [lcm_val // y for y in others]
        if _is_subset_sum(target_int, candidate_vals):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Start with known good solution
    best_set = set(range(13, 25))
    best_size = 12
    
    # Verify it's valid
    if not _is_valid_set(best_set):
        # Fallback
        best_set = set()
        best_size = 0
    
    # Try to improve by adding elements from 1-12
    # Check all subsets of [1,12] that could improve size
    # 2^12 = 4096, manageable
    
    # For each subset of [1,12], check if best_set + subset is valid
    # To be efficient, check in order of increasing size
    
    import itertools
    
    # Try adding individual elements first
    for elem in range(1, 13):
        if time.time() > deadline:
            break
        new_set = best_set | {elem}
        if len(new_set) > best_size and _is_valid_set(new_set):
            best_set = new_set
            best_size = len(best_set)
    
    # Try pairs
    for i in range(1, 13):
        if time.time() > deadline:
            break
        for j in range(i+1, 13):
            if time.time() > deadline:
                break
            new_set = best_set | {i, j}
            if len(new_set) > best_size and _is_valid_set(new_set):
                best_set = new_set
                best_size = len(best_set)
    
    # Try triples
    for i in range(1, 13):
        if time.time() > deadline:
            break
        for j in range(i+1, 13):
            if time.time() > deadline:
                break
            for k in range(j+1, 13):
                if time.time() > deadline:
                    break
                new_set = best_set | {i, j, k}
                if len(new_set) > best_size and _is_valid_set(new_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    # Try larger subsets of [1,12]
    for size in range(4, 13):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(1, 13), size):
            if time.time() > deadline:
                break
            new_set = best_set | set(subset)
            if len(new_set) > best_size and _is_valid_set(new_set):
                best_set = new_set
                best_size = len(best_set)
    
    # Also try different base sets: maybe we can drop some from [13,24] and add from [1,12]
    # This is more complex, let's try a few heuristics
    
    # Try removing one element from [13,24] and adding one or two from [1,12]
    for remove_elem in range(13, 25):
        if time.time() > deadline:
            break
        base = best_set - {remove_elem}
        for add_elem in range(1, 13):
            if time.time() > deadline:
                break
            new_set = base | {add_elem}
            if len(new_set) > best_size and _is_valid_set(new_set):
                best_set = new_set
                best_size = len(best_set)
        for i in range(1, 13):
            if time.time() > deadline:
                break
            for j in range(i+1, 13):
                if time.time() > deadline:
                    break
                new_set = base | {i, j}
                if len(new_set) > best_size and _is_valid_set(new_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
