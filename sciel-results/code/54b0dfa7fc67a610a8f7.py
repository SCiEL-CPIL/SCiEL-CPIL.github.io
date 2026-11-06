"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    selected = []
    
    # Pre-compute powers of 2 within range
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Try to add powers of 2 first
    for x in powers_of_2:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
            
    # Try to add other numbers
    # Sort candidates by some heuristic? 
    # Numbers that are "far" from existing ones might be safer.
    # Or just iterate 1..512.
    # Let's iterate 1..512 and skip those already selected.
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected:
            continue
            
        if _is_valid_addition(selected, x):
            selected.append(x)
            
    return selected

def _is_valid_addition(selected, x):
    """
    Check if adding x to selected maintains the property.
    1. x is not the mean of any subset of selected (size >= 2).
    2. No a in selected is the mean of any subset of selected + {x} (size >= 2) that includes x.
    """
    n = len(selected)
    
    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        if _subset_sum_exists(selected, x):
            return False
            
    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.
    
    for a in selected:
        # If a == x, this is handled by Check 1? No, x is distinct.
        # The subset must have size >= 2.
        # If the subset is just {x}, size 1, not allowed.
        # So we need at least one other element from selected.
        # So k >= 1.
        
        others = [y for y in selected if y != a]
        if not others:
            continue
            
        # We need to check if there exists k >= 1 such that
        # sum of k elements from 'others' is (k+1)*a - x.
        
        # Max k is len(others).
        # We can run DP on 'others'.
        # To save time, we can bound the target sum.
        # Target sum S_k = (k+1)*a - x.
        # S_k must be positive.
        # Also S_k <= sum(others).
        
        # Let's run DP for 'others' once and store results for all k?
        # Or just check specific targets.
        # Since we do this for each 'a', running DP for each 'a' is O(N * DP_cost).
        # DP_cost is O(N * MaxSum / word_size).
        # MaxSum ~ 24000. N ~ 48.
        # 48 * 48 * (24000/64) ~ 17k operations? Very fast.
        
        if _subset_sum_exists_with_target(others, a, x):
            return False
            
    return True

def _subset_sum_exists(selected, x):
    """
    Check if there is a subset of 'selected' of size k >= 2 
    such that sum(subset) == k * x.
    """
    n = len(selected)
    if n < 2:
        return False
        
    max_sum = sum(selected)
    # dp[k] is a bitset of achievable sums with k elements
    # We only care about sums up to max_sum
    # But we can just use Python integers as bitsets.
    
    dp = [0] * (n + 1)
    dp[0] = 1 # sum 0 with 0 elements
    
    for elem in selected:
        # Iterate k downwards to avoid using elem multiple times
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k+1] |= (dp[k] << elem)
                
    # Check if for any k >= 2, bit (k*x) is set in dp[k]
    for k in range(2, n + 1):
        target = k * x
        # target must be <= max_sum
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True
            
    return False

def _subset_sum_exists_with_target(others, a, x):
    """
    Check if there exists k >= 1 such that sum of k elements from 'others' 
    equals (k+1)*a - x.
    """
    n = len(others)
    if n == 0:
        return False
        
    max_sum = sum(others)
    
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for elem in others:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k+1] |= (dp[k] << elem)
                
    for k in range(1, n + 1):
        target = (k + 1) * a - x
        if target < 0:
            continue
        if target > max_sum:
            continue
        if (dp[k] >> target) & 1:
            return True
            
    return False
# EVOLVE-BLOCK-END
