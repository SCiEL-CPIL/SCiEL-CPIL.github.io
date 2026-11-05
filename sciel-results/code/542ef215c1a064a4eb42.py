"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _subset_sum_exists_mitm(values, target):
    """Check if any non-empty subset of values sums to target using meet-in-the-middle."""
    n = len(values)
    if n == 0:
        return target == 0
    
    # Quick checks
    total = sum(values)
    if total < target:
        return False
    if target == 0:
        return False  # We need non-empty subset, and all values are positive
    
    # If target equals any single element
    for v in values:
        if v == target:
            return True
    
    # Meet in the middle
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for i in range(1 << len(left)):
        s = 0
        for j in range(len(left)):
            if i & (1 << j):
                s += left[j]
        left_sums.add(s)
    
    # For each subset sum of right half, check if target - that sum is in left_sums
    for i in range(1 << len(right)):
        s_right = 0
        for j in range(len(right)):
            if i & (1 << j):
                s_right += right[j]
        
        # Check if we can form target - s_right from left half
        # But we need non-empty subset overall
        if s_right == target:
            # Right subset alone forms target (non-empty since i > 0)
            return True
        if s_right < target and (target - s_right) in left_sums:
            # Need to ensure the combined subset is non-empty
            # If s_right == 0, we need left part to be non-empty
            # If s_right > 0, subset is already non-empty
            if s_right > 0:
                return True
            else:
                # s_right == 0, need left part to be non-empty
                # Check if target can be formed by non-empty subset of left
                # We already know target != 0 (checked above)
                # And we check if target is in left_sums, but that includes empty subset (sum=0)
                # Since target > 0 and target != 0, if target in left_sums, it must be from non-empty subset
                if target in left_sums:
                    return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    # Greedy: try elements from 24 down to 1
    # When adding element x, check if 1/x can be formed by subset of reciprocals of already-selected elements
    # Already-selected elements are all > x, so their reciprocals are all < 1/x
    
    selected = []  # list of selected elements (will be in descending order)
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        # Check: can 1/x be formed as sum of reciprocals of a non-empty subset of 'selected'?
        # Convert to integer arithmetic: multiply everything by LCM of all denominators
        if not selected:
            selected.append(x)
            continue
        
        # Compute LCM of selected and x
        L = _lcm_list(selected)
        L2 = _lcm(L, x)
        
        # Target in integer terms: L2 // x
        target = L2 // x
        
        # Each selected element y contributes L2 // y
        # We need to check if any non-empty subset sums to target
        values = [L2 // y for y in selected]
        
        # Quick check: if sum of all < target, impossible
        if sum(values) < target:
            selected.append(x)
            continue
        
        # Use meet-in-the-middle for subset sum check
        if not _subset_sum_exists_mitm(values, target):
            selected.append(x)
    
    # Return as a list (order doesn't matter for the contract)
    if not selected:
        return [24]
    return list(selected)

# EVOLVE-BLOCK-END
