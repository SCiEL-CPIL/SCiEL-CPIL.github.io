"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Prime factorization of 12!
    # 2^10, 3^5, 5^2, 7^1, 11^1
    
    best = [target]  # fallback: single factor
    best_max = target
    
    # We'll do DFS building factors in strictly increasing order
    # State: (remaining_product, min_next_factor, current_factors_list, current_max)
    # But we need to track prime exponents used to ensure we can form valid factors
    
    # Alternative: enumerate all divisors of target that are > 12, sort them,
    # then do subset-sum-like search for product = target with distinct elements
    
    # Let's generate all divisors of target
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)
    
    all_divs = get_divisors(target)
    # Filter: > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # For DFS, we want to pick a subset of valid_divs (distinct) whose product is target
    # Minimize the maximum element in the subset
    # Size 1 to 12
    
    # Sort valid_divs
    valid_divs.sort()
    
    # DFS with pruning
    # We'll try to find a combination with max < current best_max
    # Use recursive search
    
    import sys
    sys.setrecursionlimit(10000)
    
    # To make this efficient, we'll use the fact that factors must be in increasing order
    # and their product must equal target
    
    # Key insight: if we pick factors in increasing order f1 < f2 < ... < fk,
    # then f1 * f2 * ... * fk = target
    # Also each fi > 12 and distinct
    
    # For pruning: if we've picked factors with product p_so_far, remaining = target // p_so_far
    # The next factor must be > last factor, and the product of remaining factors must equal remaining
    # Also, if we want k more factors all > last_factor, their product >= (last_factor+1)^(k)
    # But we don't know k in advance...
    
    # Better approach: DFS where at each step we choose the next factor from valid_divs
    # that is > last_factor, divides remaining, and < best_max
    # Base case: remaining == 1, we have a valid solution
    
    # Additional pruning: 
    # - If remaining > 1, we need at least one more factor
    # - The next factor must be > last_factor and divide remaining
    # - If remaining < (last_factor + 1), no valid next factor exists
    # - If we already have 12 factors, we can't add more
    
    found_solutions = []
    
    def dfs(remaining, min_factor, current_factors, depth):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid solution
            if current_factors:
                cur_max = max(current_factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best = list(current_factors)
                    found_solutions.append((cur_max, list(current_factors)))
            return
        
        if len(current_factors) >= 12:
            return
        
        # Pruning: if remaining < min_factor, we can't form any more factors
        if remaining < min_factor:
            return
        
        # If remaining is prime and remaining < min_factor, impossible
        # Actually, remaining must be factorable into factors all >= min_factor
        
        # The next factor must be >= min_factor, divide remaining, and < best_max
        # Also, after picking next factor f, remaining//f must be factorable into factors >= f+1 (or >= f if we allow equal? No, distinct means strictly increasing)
        
        # Wait, distinct means all factors are different. If we enforce strictly increasing order, that ensures distinctness.
        
        # So next factor f must satisfy:
        #   f >= min_factor
        #   f divides remaining
        #   f < best_max (to improve)
        #   remaining // f can be factored into distinct factors all > f (or remaining//f == 1)
        
        # For efficiency, iterate through possible factors
        # Get divisors of remaining that are >= min_factor and < best_max
        
        # To avoid iterating too many, note that f must divide remaining
        # Let's find all divisors of remaining in range [min_factor, best_max)
        
        # But computing divisors at each step is expensive. 
        # Alternative: precompute or use trial division
        
        # Let's try a different approach: since the number of prime factors is small,
        # we can enumerate all possible factor values from the prime factorization
        
        # Actually, let's just do trial division for divisors of remaining
        # that are >= min_factor
        
        r = remaining
        # Find all divisors d of r such that min_factor <= d < best_max
        # We can do this by iterating up to sqrt(r)
        
        divisors_found = []
        i = min_factor
        while i * i <= r:
            if r % i == 0:
                if i >= min_factor and i < best_max:
                    divisors_found.append(i)
                other = r // i
                if other >= min_factor and other < best_max and other != i:
                    divisors_found.append(other)
            i += 1
        
        # Also check if r itself is a valid divisor
        if r >= min_factor and r < best_max:
            divisors_found.append(r)
        
        # Sort divisors to explore smaller ones first (might find good solutions early)
        divisors_found = sorted(set(divisors_found))
        
        for f in divisors_found:
            if time.time() > deadline:
                return
            
            new_remaining = r // f
            # The next factors must all be > f (strictly increasing)
            # So min_next = f + 1
            
            # Pruning: if new_remaining > 1, we need at least one more factor > f
            # The smallest possible product of remaining factors is (f+1) if just one more
            # Actually, new_remaining must be factorable into distinct factors all > f
            # Minimum product for one more factor is f+1, so new_remaining >= f+1
            # But also new_remaining itself must be >= f+1 (since the next factor divides it and is >= f+1)
            
            if new_remaining > 1:
                # We need at least one more factor, which must be > f
                # So new_remaining must be >= f + 1 (trivially true since new_remaining >= 1 and f >= 13)
                # More importantly, new_remaining must have a divisor >= f+1
                # The largest proper divisor of new_remaining is new_remaining itself
                # So we need new_remaining >= f + 1
                if new_remaining < f + 1:
                    continue
            
            # Additional pruning: the maximum number of additional factors we can have
            # is limited by 12 - len(current_factors) - 1
            # Each additional factor is at least f+1, so product >= (f+1)^count
            # But this is a weak bound
            
            current_factors.append(f)
            dfs(new_remaining, f + 1, current_factors, depth + 1)
            current_factors.pop()
    
    dfs(target, 13, [], 0)
    
    if best_max < target:
        return best
    else:
        return [target]

# EVOLVE-BLOCK-END
