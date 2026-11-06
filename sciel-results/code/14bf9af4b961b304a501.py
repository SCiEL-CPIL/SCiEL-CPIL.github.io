"""Optimized solver for sum of distinct divisors of 5040 equal to 4979."""
import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Filter: we can only use divisors <= target
    divisors = [d for d in divisors if d <= target]
    d_count = len(divisors)
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (d_count + 1)
    for i in range(d_count - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Best solution found so far
    best_count = d_count + 1
    best_solution = None
    
    # Try to find solution with k terms, starting from small k
    # Lower bound on k: target / max_divisor
    max_div = divisors[0]
    min_k = (target + max_div - 1) // max_div
    
    # We'll use DFS with pruning
    # State: index in divisors, remaining sum, current count
    # Pruning: 
    #   1. If current count >= best_count, prune
    #   2. If suffix_sum[index] < remaining, prune
    #   3. If remaining == 0, record solution
    #   4. If index == d_count, prune
    
    def dfs(index, remaining, count, chosen):
        nonlocal best_count, best_solution, deadline
        
        if time.time() > deadline:
            return False  # timeout
        
        if remaining == 0:
            if count < best_count:
                best_count = count
                best_solution = chosen[:]
            return True
        
        if index >= d_count:
            return False
        
        if count >= best_count:
            return False
        
        if suffix_sum[index] < remaining:
            return False
        
        # Try to find a solution with exactly (best_count - count - 1) more terms
        # This helps prune: if we need more terms than available, prune
        
        # Try including divisors[index]
        d = divisors[index]
        
        # Option 1: include this divisor
        if d <= remaining:
            chosen.append(d)
            # Check if we can complete with remaining terms
            # Need (best_count - count - 1) more terms from remaining divisors
            if dfs(index + 1, remaining - d, count + 1, chosen):
                chosen.pop()
                return True
            chosen.pop()
        
        # Option 2: skip this divisor
        # But we need to ensure we can still reach remaining with the remaining divisors
        # and with at most (best_count - count - 1) more terms
        if dfs(index + 1, remaining, count, chosen):
            return True
        
        return False
    
    # Try increasing k from min_k
    for k in range(min_k, d_count + 1):
        if time.time() > deadline:
            break
        
        # Reset best_count to k for this search level
        # We want to find a solution with exactly k terms
        # Use a different approach: search for solution with at most k terms
        
        best_count = k + 1  # So that solutions with k terms are accepted
        best_solution = None
        chosen = []
        
        # Modified DFS that stops at k terms
        def dfs_k(index, remaining, count, chosen):
            nonlocal best_count, best_solution, deadline
            
            if time.time() > deadline:
                return False
            
            if remaining == 0:
                if count <= k:
                    best_count = count
                    best_solution = chosen[:]
                return True
            
            if index >= d_count:
                return False
            
            if count >= k:
                return False
            
            if suffix_sum[index] < remaining:
                return False
            
            # Additional pruning: we need at least (remaining / divisors[index]) more terms
            # since divisors are sorted descending, max possible per term is divisors[index]
            min_terms_needed = (remaining + divisors[index] - 1) // divisors[index]
            if count + min_terms_needed > k:
                return False
            
            # Also, we can't use more than (k - count) terms, and the sum of the largest
            # (k - count) remaining divisors must be >= remaining
            # This is a stronger bound
            
            d = divisors[index]
            
            # Option 1: include this divisor
            if d <= remaining:
                chosen.append(d)
                if dfs_k(index + 1, remaining - d, count + 1, chosen):
                    chosen.pop()
                    return True
                chosen.pop()
            
            # Option 2: skip this divisor
            if dfs_k(index + 1, remaining, count, chosen):
                return True
            
            return False
        
        if dfs_k(0, target, 0, chosen):
            if best_solution is not None:
                return best_solution
        else:
            # Timeout or no solution found for this k
            if time.time() > deadline:
                if best_solution is not None:
                    return best_solution
                # Return a valid solution (the greedy one) as fallback
                remainder = target
                result = []
                for d in divisors:
                    if d <= remainder:
                        result.append(d)
                        remainder -= d
                if remainder == 0:
                    return result
                # Should not happen, but just in case
                break
    
    # Fallback: greedy solution
    if best_solution is not None:
        return best_solution
    
    remainder = target
    result = []
    for d in divisors:
        if d <= remainder:
            result.append(d)
            remainder -= d
    return result
