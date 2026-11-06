"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    selected = []
    
    # Strategic ordering: start with well-separated numbers
    # Powers of 2 have good separation properties
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # First pass: add powers of 2 (fast, low risk)
    for x in powers_of_2:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
    
    # Second pass: try remaining numbers in increasing order
    # Numbers that are far from existing elements are safer
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
    Check if adding x to selected maintains the no-mean property.
    1. x is not the mean of any subset of selected (size >= 2).
    2. No a in selected is the mean of any subset of selected + {x} (size >= 2) that includes x.
    """
    n = len(selected)
    
    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        if _subset_mean_exists(selected, x):
            return False
            
    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.
    
    for a in selected:
        others = [y for y in selected if y != a]
        if not others:
            continue
            
        # We need to check if there exists k >= 1 such that
        # sum of k elements from 'others' is (k+1)*a - x.
        
        if _subset_sum_with_target_exists(others, a, x):
            return False
            
    return True

def _subset_mean_exists(selected, x):
    """
    Check if there is a subset of 'selected' of size k >= 2 
    such that sum(subset) == k * x.
    """
    n = len(selected)
    if n < 2:
        return False
        
    max_sum = sum(selected)
    
    # dp[k] is a bitset of achievable sums with k elements
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for elem in selected:
        # Iterate k downwards to avoid using elem multiple times
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k+1] |= (dp[k] << elem)
                
    # Check if for any k >= 2, bit (k*x) is set in dp[k]
    for k in range(2, n + 1):
        target = k * x
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True
            
    return False

def _subset_sum_with_target_exists(others, a, x):
    """
    Check if there exists k >= 1 such that sum of k elements from 'others' 
    equals (k+1)*a - x.
    """
    n = len(others)
    if n == 0:
        return False
        
    max_sum = sum(others)
    
    # dp[k] is a bitset of achievable sums with k elements
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
