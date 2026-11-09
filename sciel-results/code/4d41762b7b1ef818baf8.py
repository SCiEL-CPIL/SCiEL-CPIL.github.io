import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Weights
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending to improve B&B pruning
    # We need to map back to original indices
    indexed_weights = [(weights[i], i) for i in range(n)]
    indexed_weights.sort(reverse=True, key=lambda x: x[0])
    sorted_weights = [w for w, _ in indexed_weights]
    sorted_orig_indices = [i for _, i in indexed_weights]
    
    # Suffix sums for upper bound
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    best_sum = 0
    best_mask = 0
    
    # Greedy initial solution
    curr_sum = 0
    curr_mask = 0
    for i in range(n):
        if curr_sum + sorted_weights[i] < L:
            curr_sum += sorted_weights[i]
            curr_mask |= (1 << i)
    best_sum = curr_sum
    best_mask = curr_mask
    
    # Branch and Bound
    def search(i, curr_sum, curr_mask):
        nonlocal best_sum, best_mask
        if time.time() - start_time > time_limit:
            return
        if i == n:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_mask = curr_mask
            return
            
        # Pruning
        if curr_sum + suffix_sums[i] <= best_sum:
            return
            
        # Option 1: Include item i
        w_i = sorted_weights[i]
        if curr_sum + w_i < L:
            search(i + 1, curr_sum + w_i, curr_mask | (1 << i))
            
        # Option 2: Exclude item i
        search(i + 1, curr_sum, curr_mask)
        
    search(0, 0, 0)
    
    # Convert mask back to original indices
    selected_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            selected_indices.append(sorted_orig_indices[i])
            
    return selected_indices
