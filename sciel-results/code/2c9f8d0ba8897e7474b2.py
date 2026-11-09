import time
import random
import math
import sys

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    # Primes powers: 2^4, 3^3, 5^2, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    L = 1
    L *= 16
    L *= 27
    L *= 25
    L *= 7
    L *= 11
    L *= 13
    L *= 17
    L *= 19
    L *= 23
    L *= 29
    L *= 31
    L *= 37
    L *= 41
    L *= 43
    L *= 47
    
    weights = [L // d for d in denominators]
    target = L - 1
    
    # Sort items by weight descending to find high-value solutions early
    # Index i corresponds to denominator denominators[i]
    # We need to return the original indices
    items = list(zip(weights, range(n)))
    items.sort(key=lambda x: x[0], reverse=True)
    sorted_weights = [w for w, _ in items]
    sorted_indices = [idx for _, idx in items]
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of sorted_weights[i:]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]
    
    # 1. Greedy Initialization
    curr_sum = 0
    curr_set = set()
    for i in range(n):
        if curr_sum + sorted_weights[i] <= target:
            curr_sum += sorted_weights[i]
            curr_set.add(sorted_indices[i])
    
    best_sum = curr_sum
    best_set = set(curr_set)
    
    start_time = time.time()
    time_limit = 50.0 # Safety margin
    
    # Increase recursion limit just in case, though 47 is small
    sys.setrecursionlimit(10000)
    
    # 2. Branch and Bound Search
    def search(idx, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
        
        # Prune: if current_sum + sum of all remaining <= best_sum, no point continuing
        if current_sum + suffix_sum[idx] <= best_sum:
            return True
        
        # If we've considered all items
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return True
            
        # Option 1: Include item idx
        new_sum = current_sum + sorted_weights[idx]
        if new_sum <= target:
            current_set.append(sorted_indices[idx])
            
            # If this path is already better, update best
            if new_sum > best_sum:
                best_sum = new_sum
                best_set = set(current_set)
            
            # Recurse to include
            include_ok = search(idx + 1, new_sum, current_set)
            current_set.pop()
            
            if not include_ok:
                return False
        
        # Option 2: Exclude item idx
        # We only need to check if excluding could still lead to a better sum.
        # The top-level check `current_sum + suffix_sum[idx] <= best_sum` ensures that
        # *some* combination of idx..n-1 can beat best_sum.
        # If including idx didn't work (or was pruned), we try excluding.
        
        return search(idx + 1, current_sum, current_set)
    
    # Start search
    search(0, 0, [])
    
    return list(best_set)
