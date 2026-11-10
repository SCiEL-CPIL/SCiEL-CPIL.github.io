"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
from itertools import combinations

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_single_violation(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some nonempty subset of others."""
    if not others:
        return False
    
    # Use exact integer arithmetic
    # 1/target_x = sum(1/y for y in subset)
    # Multiply by LCM of [target_x] + others:
    # LCM/target_x = sum(LCM/y for y in subset)
    
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    
    # Check if target_int can be formed as subset sum of candidate_vals
    n = len(candidate_vals)
    if n == 0:
        return False
    if n <= 20:
        # Meet in the middle
        half = n // 2
        left = candidate_vals[:half]
        right = candidate_vals[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # For each subset sum of right, check if target - sum is in left_sums
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if (target_int - s) in left_sums:
                return True
        return False
    else:
        # For larger sets, use a simpler check - just try all subsets if n is small enough
        # Otherwise, this might be slow, but for n up to 23 it's manageable with pruning
        # Use a different approach: check if target_int is in the set of all subset sums
        # Use a boolean DP if values are small enough, otherwise brute force with pruning
        
        # Try all subsets with pruning
        # Sort candidate_vals
        sorted_vals = sorted(candidate_vals)
        
        # Quick check: if sum of all < target_int, return False
        if sum(sorted_vals) < target_int:
            return False
        if sorted_vals[0] > target_int:
            return False
        
        # Use meet in the middle for n up to ~24
        # Actually for n=23, 2^23 is too large. Let's use a smarter approach.
        # For our problem, n is at most 23 (24 elements minus 1 target)
        # 2^23 = 8.4 million, which is feasible in Python if optimized
        
        # Let's just do meet in the middle for any n
        half = n // 2
        left = candidate_vals[:half]
        right = candidate_vals[half:]
        
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if (target_int - s) in left_sums:
                return True
        return False

def _is_valid_set(S):
    """Check if set S satisfies the reciprocal-subsum-free property."""
    S_list = list(S)
    for x in S_list:
        others = [y for y in S_list if y != x]
        if _check_single_violation(x, others):
            return False
    return True

def _count_violations(S):
    """Count how many elements in S have a violation."""
    S_list = list(S)
    count = 0
    for x in S_list:
        others = [y for y in S_list if y != x]
        if _check_single_violation(x, others):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Strategy: Start with all 24 elements, iteratively remove the element
    # whose removal fixes the most violations.
    
    S = set(range(1, 25))
    
    # First, check if the full set is valid
    if _is_valid_set(S):
        return list(S)
    
    # Iterative removal
    while True:
        if time.time() > deadline:
            break
        
        if _is_valid_set(S):
            break
        
        # Find which element to remove
        best_removal = None
        best_score = -1
        
        for x in S:
            if time.time() > deadline:
                break
            S_without_x = S - {x}
            # Count violations in S_without_x
            viol = _count_violations(S_without_x)
            # We want to minimize violations, so score = -viol
            score = -viol
            if score > best_score:
                best_score = score
                best_removal = x
        
        if best_removal is None:
            break
        
        S.discard(best_removal)
    
    # Try to add back elements if possible
    for x in range(1, 25):
        if time.time() > deadline:
            break
        if x not in S:
            S_trial = S | {x}
            if _is_valid_set(S_trial):
                S = S_trial
    
    # Final check
    if not S:
        S = {24}  # Fallback: single element is always valid
    
    return list(S)
# EVOLVE-BLOCK-END
