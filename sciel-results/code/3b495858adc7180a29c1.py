"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Find all divisors of N that are > 12
    divisors = []
    for i in range(13, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
    # Also need to check if N itself > 12 (it is)
    if N > 12:
        divisors.append(N)
    divisors = sorted(set(divisors))
    
    # For efficiency, we want to search for small max factor
    # Strategy: binary search on the maximum allowed factor
    # For a given max_factor M, can we find distinct factors all in (12, M] with product N?
    
    # Precompute: for each divisor d > 12, d must be <= M
    # Use DFS with pruning
    
    best_max = N  # worst case: single factor
    best_factors = [N]
    
    # Sort divisors for DFS
    # We want to find a set of distinct divisors > 12, product = N, minimize max
    
    # Approach: DFS choosing factors in increasing order to ensure distinctness
    # and prune when current choice exceeds best_max
    
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid decomposition
            if current_factors:
                m = current_max
                if m < best_max:
                    best_max = m
                    best_factors = list(current_factors)
            return
        
        # Pruning: if remaining is a divisor > 12 and not in current_factors
        if remaining > 12 and remaining not in current_factors:
            if len(current_factors) + 1 <= 12:
                m = max(current_max, remaining)
                if m < best_max:
                    best_max = m
                    best_factors = list(current_factors) + [remaining]
            return
        
        # Need at least one more factor
        # The smallest possible next factor is divisors[start_idx] if start_idx < len(divisors)
        if start_idx >= len(divisors):
            return
        
        # Pruning: if the product of the smallest k remaining divisors (k factors) 
        # can't reach 'remaining', we can't complete
        # Actually, let's just iterate through possible next factors
        
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            
            if time.time() > deadline:
                return
            
            # d must divide remaining
            if remaining % d != 0:
                continue
            
            # d must be > 12 (already ensured by construction)
            # d must not be in current_factors (ensured by start_idx since we go in order)
            
            new_max = max(current_max, d)
            
            # Pruning: if new_max >= best_max, no improvement possible
            if new_max >= best_max:
                # Since divisors are sorted, all subsequent d' >= d, so new_max' >= new_max >= best_max
                break
            
            new_remaining = remaining // d
            new_factors = current_factors + [d]
            
            if len(new_factors) > 12:
                break
            
            dfs(new_remaining, i + 1, new_factors, new_max)
    
    # Initial call
    dfs(N, 0, [], 12)
    
    return best_factors

# EVOLVE-BLOCK-END
