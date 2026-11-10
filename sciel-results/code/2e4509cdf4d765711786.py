"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some subset of others."""
    if not others:
        return False
    # Convert to integer arithmetic: 1/target_x = sum(1/y for y in subset)
    # Multiply both sides by LCM of [target_x] + others
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    
    # Check if target_int can be formed as subset sum of candidate_vals
    # Use meet-in-the-middle for efficiency
    n = len(candidate_vals)
    if n <= 20:
        # Simple subset sum check
        possible = {0}
        for v in candidate_vals:
            new_possible = set()
            for s in possible:
                new_possible.add(s)
                new_possible.add(s + v)
            possible = new_possible
            if target_int in possible:
                return True
        return target_int in possible
    else:
        # Meet in the middle
        half = n // 2
        left_vals = candidate_vals[:half]
        right_vals = candidate_vals[half:]
        
        left_sums = {0}
        for v in left_vals:
            new_left = set()
            for s in left_sums:
                new_left.add(s)
                new_left.add(s + v)
            left_sums = new_left
        
        right_sums = {0}
        for v in right_vals:
            new_right = set()
            for s in right_sums:
                new_right.add(s)
                new_right.add(s + v)
            right_sums = new_right
        
        for ls in left_sums:
            if target_int - ls in right_sums:
                return True
        return False

def _is_valid_set(selected):
    """Check if the selected set is reciprocal-subsum-free."""
    for i, x in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _branch_and_bound(remaining, current, best, deadline):
    """Branch and bound to find maximum valid subset."""
    if time.time() > deadline:
        return best
    
    if len(current) + len(remaining) <= len(best):
        return best
    
    if not remaining:
        if len(current) > len(best):
            best = current[:]
        return best
    
    # Try including the first element
    x = remaining[0]
    rest = remaining[1:]
    
    # Check if adding x to current would create a conflict
    # We need to check: does 1/x equal sum of reciprocals of a subset of (current + rest)?
    # And for each y in current, does 1/y equal sum of reciprocals of a subset of (current - {y} + [x] + rest)?
    
    # First check: is 1/x a subset sum of reciprocals of current?
    if _check_reciprocal_subsum(x, current):
        # Can't add x, skip it
        return _branch_and_bound(rest, current, best, deadline)
    
    # Second check: for each y in current, does 1/y become a subset sum of (current - {y}) + [x]?
    can_add = True
    for y in current:
        others_for_y = [z for z in current if z != y] + [x]
        if _check_reciprocal_subsum(y, others_for_y):
            can_add = False
            break
    
    if can_add:
        new_current = current + [x]
        best = _branch_and_bound(rest, new_current, best, deadline)
    
    # Try excluding x
    best = _branch_and_bound(rest, current, best, deadline)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Start with a greedy solution as initial best
    # Descending order greedy: add elements from large to small
    candidates = list(range(24, 0, -1))
    current = []
    for x in candidates:
        if time.time() > deadline:
            break
        # Check if adding x to current keeps the set valid
        if _check_reciprocal_subsum(x, current):
            continue
        can_add = True
        for y in current:
            others_for_y = [z for z in current if z != y] + [x]
            if _check_reciprocal_subsum(y, others_for_y):
                can_add = False
                break
        if can_add:
            current.append(x)
    
    best = current[:]
    
    # Now try to improve with branch and bound
    # Sort candidates in descending order for better pruning
    all_candidates = list(range(24, 0, -1))
    
    # Use branch and bound starting from empty set
    best = _branch_and_bound(all_candidates, [], best, deadline)
    
    return best
# EVOLVE-BLOCK-END
