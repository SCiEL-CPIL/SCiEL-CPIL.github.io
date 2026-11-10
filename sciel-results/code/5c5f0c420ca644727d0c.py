"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = lst[0]
    for v in lst[1:]:
        result = _lcm(result, v)
    return result

def _can_form_target(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals.
    vals is a list of integers (scaled reciprocals).
    Uses meet-in-the-middle for efficiency.
    """
    if target == 0:
        return True
    if not vals:
        return False
    if target > sum(vals):
        return False
    
    n = len(vals)
    if n <= 20:
        # Use meet-in-the-middle
        half = n // 2
        left = vals[:half]
        right = vals[half:]
        
        left_sums = {0}
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        
        right_sums = {0}
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}
        
        for s in left_sums:
            if target - s in right_sums:
                return True
        return False
    else:
        # For larger sets, use a simpler check or approximation
        # But since our sets are at most 24 elements, and we're checking
        # subsets of OTHER elements, this should be fine
        # Actually, let's just use the same approach but it might be slow
        # Let's try a different approach: sort and use DP if target is small
        # But target can be large. Let's stick with meet-in-the-middle
        # but cap the size
        if n > 24:
            # This shouldn't happen in our problem
            return False
        half = n // 2
        left = vals[:half]
        right = vals[half:]
        
        left_sums = {0}
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        
        right_sums = {0}
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}
        
        for s in left_sums:
            if target - s in right_sums:
                return True
        return False

def _is_valid_set(selected):
    """
    Check if the set of selected elements is reciprocal-subsum-free.
    For each element x in selected, 1/x should NOT equal the sum of
    reciprocals of any subset of the OTHER selected elements.
    """
    if len(selected) <= 1:
        return True
    
    # Scale all reciprocals by the LCM of all elements
    lcm_val = _lcm_list(selected)
    
    for i, x in enumerate(selected):
        # Target: 1/x scaled by lcm_val = lcm_val / x
        target = lcm_val // x
        
        # Other elements: selected without x
        others = selected[:i] + selected[i+1:]
        if not others:
            continue
        
        # Scaled values of others
        others_scaled = [lcm_val // y for y in others]
        
        if _can_form_target(target, others_scaled):
            return False
    
    return True

def _try_add_element(current_set, candidate, deadline):
    """
    Try to add candidate to current_set. Returns True if valid.
    """
    new_set = current_set + [candidate]
    # Only need to check if adding candidate creates a violation
    # This means:
    # 1. 1/candidate should not be sum of reciprocals of subset of current_set
    # 2. For each x in current_set, 1/x should not be sum of reciprocals of 
    #    subset of (current_set - {x}) ∪ {candidate}
    #    But we already know current_set is valid, so we only need to check
    #    if adding candidate creates NEW violations
    
    # Check 1: 1/candidate vs current_set
    lcm_current = _lcm_list(current_set)
    target1 = lcm_current // candidate
    others1_scaled = [lcm_current // y for y in current_set]
    if _can_form_target(target1, others1_scaled):
        return False
    
    # Check 2: For each x in current_set, check if 1/x can now be formed
    # using a subset that includes candidate
    # The new subsets that include candidate are: {candidate} ∪ S where S ⊆ current_set - {x}
    # So we need to check if 1/x - 1/candidate = sum of reciprocals of S
    # i.e., (candidate - x) / (x * candidate) = sum of reciprocals of S
    # Scaled by LCM of (current_set ∪ {candidate}):
    
    lcm_new = _lcm(lcm_current, candidate)
    
    for x in current_set:
        # Target: 1/x - 1/candidate = (candidate - x) / (x * candidate)
        # Scaled by lcm_new: lcm_new * (candidate - x) / (x * candidate)
        # = lcm_new / (x * candidate) * (candidate - x)
        # Let's compute: lcm_new * (candidate - x) / (x * candidate)
        
        # For this to be an integer, x * candidate must divide lcm_new * (candidate - x)
        # Since lcm_new is a multiple of both x and candidate, lcm_new / x and 
        # lcm_new / candidate are integers
        # lcm_new * (candidate - x) / (x * candidate) = (lcm_new / x) * (candidate - x) / candidate
        # This is integer iff candidate divides (lcm_new / x) * (candidate - x)
        # Since candidate divides lcm_new, and candidate - x may or may not...
        # Let's just compute it as a rational
        
        numerator = lcm_new * (candidate - x)
        denominator = x * candidate
        
        if numerator % denominator != 0:
            # Not an integer, so can't be a sum of integer scaled reciprocals
            # Wait, the scaled reciprocals are integers, so their sum is an integer
            # So if the target isn't an integer, it can't be formed
            continue
        
        target2 = numerator // denominator
        
        # Other elements: current_set - {x}
        others2 = [y for y in current_set if y != x]
        if not others2:
            # Only candidate is available, but we're checking if 1/x - 1/candidate = 0
            # which means x = candidate, but they're distinct
            if target2 == 0:
                # This would mean 1/x = 1/candidate, i.e., x = candidate
                # But they're distinct, so this shouldn't happen
                return False
            continue
        
        others2_scaled = [lcm_new // y for y in others2]
        
        if _can_form_target(target2, others2_scaled):
            return False
    
    return True

def construct():
    deadline = time.time() + 85  # 5 second safety margin
    
    best_set = list(range(13, 25))  # Start with the seed
    best_size = len(best_set)
    
    # Try to improve by adding elements
    # Use a greedy approach with random restarts
    
    all_elements = list(range(1, 25))
    
    # First, try a greedy approach starting from different initial sets
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Randomly shuffle the elements
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        
        current_set = []
        
        for elem in shuffled:
            if time.time() > deadline:
                break
            
            if _try_add_element(current_set, elem, deadline):
                current_set.append(elem)
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
        
        # Also try to add elements to the current best set
        if len(current_set) == best_size:
            # Try to improve by swapping
            pass
    
    # Try to add small elements to the seed
    seed_set = list(range(13, 25))
    for small in range(1, 13):
        if time.time() > deadline:
            break
        if _try_add_element(seed_set, small, deadline):
            seed_set.append(small)
    
    if len(seed_set) > best_size:
        best_set = seed_set[:]
        best_size = len(seed_set)
    
    return best_set

# EVOLVE-BLOCK-END
