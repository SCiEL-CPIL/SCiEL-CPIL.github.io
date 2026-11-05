import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline for 90s limit, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators, 1)
    
    # Calculate integer weights w_i = L / d_i
    weights = [L // d for d in denominators]
    
    # Sort items by weight descending (denominator ascending)
    # We need to keep track of original indices
    items = sorted(zip(weights, range(n)), key=lambda x: -x[0])
    sorted_weights = [w for w, _ in items]
    sorted_indices = [idx for _, idx in items]
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of sorted_weights[i:]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]
    
    best_sum = 0
    best_indices = []
    
    # Initial greedy solution to set a strong lower bound
    curr_sum = 0
    curr_indices = []
    for i in range(n):
        if curr_sum + sorted_weights[i] < L:
            curr_sum += sorted_weights[i]
            curr_indices.append(sorted_indices[i])
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_indices = curr_indices[:]
        
    # DFS Branch and Bound with in-place list modification
    current_path = []
    
    def dfs(idx, curr_sum):
        nonlocal best_sum, best_indices
        nonlocal current_path
        
        # Time check
        if time.time() > deadline:
            return False # Signal timeout
            
        # Pruning 1: If we can't beat the best sum even by taking all remaining items
        if curr_sum + suffix_sum[idx] <= best_sum:
            return True
            
        # Pruning 2: If current sum is already >= L, we can't add more
        if curr_sum >= L:
            return True
            
        # Base case: reached end of list
        if idx == n:
            if curr_sum < L and curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = current_path[:]
            return True
            
        # Option 1: Include current item
        # Only if adding it keeps sum < L
        if curr_sum + sorted_weights[idx] < L:
            current_path.append(sorted_indices[idx])
            if not dfs(idx + 1, curr_sum + sorted_weights[idx]):
                return False
            current_path.pop()
            
        # Option 2: Exclude current item
        # Check if it's worth excluding
        # If excluding it, the max possible sum is curr_sum + suffix_sum[idx+1]
        # If that is <= best_sum, we can prune this branch too.
        if curr_sum + suffix_sum[idx + 1] > best_sum:
            if not dfs(idx + 1, curr_sum):
                return False
                
        return True

    # Run DFS
    # Start with empty set
    dfs(0, 0)
    
    # Return the best indices found
    # Ensure non-empty
    if not best_indices:
        # Fallback: just pick index 0 (denom 2, sum 0.5)
        return [0]
        
    return best_indices
