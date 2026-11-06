"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def construct():
    target = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Get prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7
    # We need to find distinct integers > 12 whose product is 12!
    # Minimize the largest factor
    
    # Strategy: enumerate all divisors of 12! that are > 12, then search
    # for subsets of distinct divisors whose product is 12!
    
    # First, get all divisors of 12!
    def get_all_divisors(n):
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divs = get_all_divisors(target)
    # Filter to divisors > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # We want to find a subset of valid_divs (distinct, size 1-12) 
    # whose product is exactly target, minimizing the maximum element.
    
    # Sort valid_divs to help with pruning
    valid_divs.sort()
    
    best_factors = None
    best_max = float('inf')
    
    # Use DFS with pruning
    # State: index in valid_divs, remaining product, current max, current factors list
    # We can also use the fact that if we pick a divisor d, remaining must be divisible by d
    
    # Precompute: for each divisor d in valid_divs, check if target % d == 0
    # (they all are by construction)
    
    # Key insight: we can use meet-in-the-middle or smart DFS
    # Since 12! has a manageable number of divisors, let's count them
    # Number of divisors of 12! = (10+1)*(5+1)*(2+1)*(1+1) = 11*6*3*2 = 396
    # So valid_divs has at most 396 - 12 = 384 elements (excluding divisors <= 12)
    
    # Let's use a recursive search with memoization-like pruning
    # Sort valid_divs in descending order to try larger values first? 
    # No, we want to minimize the max, so we should try smaller values first
    # but we also need to find a valid combination quickly.
    
    # Alternative: iterate over possible max values M, and check if there exists
    # a subset of valid_divs all <= M whose product is target.
    # Binary search on M!
    
    # But first, let's try a direct DFS approach with good pruning
    
    def dfs(idx, remaining, current_max, current_factors):
        nonlocal best_factors, best_max
        
        # Time check
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid decomposition
            if current_max < best_max:
                best_max = current_max
                best_factors = list(current_factors)
            return
        
        if remaining <= 12:
            # Can't use any more factors since all must be > 12
            return
        
        # Pruning: if current_max >= best_max, no need to continue
        if current_max >= best_max:
            return
        
        # Try each divisor from idx onwards
        for i in range(idx, len(valid_divs)):
            d = valid_divs[i]
            
            # If d > remaining, we can't use it (and larger ones won't work either)
            if d > remaining:
                break
            
            # If remaining % d != 0, skip
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                # Since valid_divs is sorted, all subsequent d' >= d will give new_max >= d >= best_max
                # Actually not quite, but this helps
                if d >= best_max:
                    break
            
            # Pruning: the minimum possible product of remaining factors
            # If we need k more factors, each at least 13, the min product is 13^k
            # But we don't know k. Instead, check if remaining/d can be factored
            # into factors all > 12 and >= d (to maintain distinctness and ordering)
            
            # Simple check: remaining/d must be >= 1 (always true)
            # And if remaining/d == 1, we're done (handled above)
            
            # Additional pruning: the smallest possible next factor is d+1 (since distinct and sorted)
            # Actually no, we're iterating in order, so next factor will be > d
            # But we need remaining/d to be factorable into factors all > 12 and > d... 
            # Wait, no. The factors just need to be distinct and > 12.
            # Since we're picking in sorted order, subsequent factors will be > d.
            
            # Key pruning: remaining/d must be >= 1
            # And if remaining/d > 1, we need at least one more factor > 12
            # The smallest such factor greater than d is at least d+1 (but actually the next divisor > d)
            # So remaining/d must be >= (next divisor after d) if we need more factors
            # This is complex. Let's just try it.
            
            current_factors.append(d)
            dfs(i + 1, remaining // d, new_max, current_factors)
            current_factors.pop()
            
            # Time check
            if time.time() > deadline:
                return
    
    # Start DFS
    dfs(0, target, 0, [])
    
    # If we found something, return it
    if best_factors is not None:
        return best_factors
    
    # Fallback: return [12!] as a single factor (valid but not optimal)
    return [target]

# EVOLVE-BLOCK-END
