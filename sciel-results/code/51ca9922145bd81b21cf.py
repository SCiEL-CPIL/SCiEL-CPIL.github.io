"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    n_val = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisor_set = set(divisors)
    divisors.sort()
    n_divs = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    total_sum = prefix[n_divs]
    
    # k=1: check if target itself is a divisor
    if target in divisor_set:
        return [target]
    
    # k=2: O(n) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement != d and complement > 0:
            return [d, complement]
    
    # k=3: O(n^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            complement = target - a - b
            if complement in divisor_set and complement > b:
                return [a, b, complement]
    
    # For k >= 4, use DP with memoization
    # dp[i][k][s] = can we form sum s using exactly k divisors from divisors[i:]?
    # Since divisors are sorted, we can use this efficiently
    
    for k in range(4, 257):
        if time.time() > deadline:
            break
            
        # Check if it's possible to sum to target with k divisors
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        
        if target < min_sum_k or target > max_sum_k:
            continue
        
        # Use DP with memoization: state is (i, remaining_k, remaining_sum)
        # We return the actual solution path when found
        
        @lru_cache(maxsize=None)
        def dp(i, remaining_k, remaining_sum):
            """Returns a tuple of chosen divisors if possible, None otherwise"""
            if remaining_k == 0:
                return () if remaining_sum == 0 else None
            
            if i >= n_divs:
                return None
            
            # Not enough elements left
            if n_divs - i < remaining_k:
                return None
            
            # Pruning: min possible sum
            if i + remaining_k <= n_divs:
                min_possible = prefix[i + remaining_k] - prefix[i]
                if remaining_sum < min_possible:
                    return None
            
            # Pruning: max possible sum from position i
            # Largest remaining_k elements from i to end
            # We can compute this more carefully
            # Actually, let's just try both branches
            
            # Try including divisors[i]
            if remaining_sum >= divisors[i]:
                result = dp(i + 1, remaining_k - 1, remaining_sum - divisors[i])
                if result is not None:
                    return (divisors[i],) + result
            
            # Try excluding divisors[i]
            result = dp(i + 1, remaining_k, remaining_sum)
            if result is not None:
                return result
            
            return None
        
        result = dp(0, k, target)
        if result is not None:
            return list(result)
        
        # Clear cache for next k
        dp.cache_clear()
    
    # Fallback: greedy (should not reach here if solution exists and time is sufficient)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    
    return chosen
# EVOLVE-BLOCK-END
