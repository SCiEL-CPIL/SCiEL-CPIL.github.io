"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = factorial(12)  # 479001600
    
    # Get all prime factorization of N
    # N = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Generate all divisors of N that are > 12
    # Use prime factorization to generate divisors
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    def gen_divisors(idx, current):
        if idx == len(primes):
            if current > 12:
                yield current
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            yield from gen_divisors(idx + 1, current * val)
            val *= p
    
    # This might be slow; let's be smarter
    # Number of divisors: (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # That's manageable
    
    all_divisors = list(gen_divisors(0, 1))
    all_divisors.sort()
    
    # For each divisor d > 12, N/d must also be a divisor
    # We want to partition N into distinct factors > 12
    # Use DFS with branch and bound
    
    best_max = [N]  # initial: single factor
    best_factors = [[N]]
    
    def dfs(remaining, min_factor, current_factors, current_max):
        # Check deadline
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a complete partition
            if current_factors:
                m = max(current_factors)
                if m < best_max[0]:
                    best_max[0] = m
                    best_factors[0] = list(current_factors)
            return
        
        # Pruning: if current_max >= best_max[0], no improvement possible
        # (we still need at least one more factor, which will be >= min_factor)
        # Actually the new factor could be smaller... but we're going in increasing order
        
        # We need to pick a divisor d of 'remaining' such that:
        #   d >= min_factor, d > 12, d divides remaining
        #   and remaining/d can be further decomposed into distinct factors >= d
        #   (or remaining/d == 1 if this is the last factor)
        
        # Get divisors of 'remaining' that are >= min_factor and > 12
        # Also d must be <= remaining (trivially)
        # And remaining/d must be achievable with factors >= d (distinct, > 12)
        
        # Lower bound on the number of more factors needed:
        # If we pick d, then remaining/d must be factored into factors >= d, > 12, distinct from current_factors
        # The minimum product of k factors each >= d is d*(d+1)*...*(d+k-1)
        # So we need d*(d+1)*...*(d+k-1) <= remaining/d for some k >= 1
        
        # Simpler pruning: if remaining < min_factor * min_factor, we can't split into 2+ factors
        # (since both would be >= min_factor)
        # Actually if remaining < min_factor, we can't even pick one more factor >= min_factor
        # But remaining must be >= min_factor since we pick d >= min_factor and d divides remaining
        
        # Let's just enumerate divisors of remaining that are >= min_factor
        # For efficiency, iterate through all_divisors and check divisibility
        
        # Optimization: only consider d where d <= remaining and d >= min_factor
        # and remaining % d == 0
        # Also, to avoid the trivial case where d == remaining (single factor left),
        # we handle that: if d == remaining, then we just add it and check if > 12
        
        # Branch and bound: the new factor d must satisfy d >= min_factor
        # And we want to keep factors balanced to minimize max
        
        # Heuristic: try factors close to the geometric mean
        # If we need k more factors, each ~ (remaining)^(1/k)
        
        # For now, just enumerate in order and prune
        
        # Upper bound on d: d must be <= remaining, and remaining/d must be >= min_factor
        # (if we're going to split further) OR remaining/d == 1 (last factor)
        # Actually remaining/d just needs to be factorable into factors >= d (which is >= min_factor)
        # The smallest such factor is d itself, so remaining/d >= d means d^2 <= remaining
        # UNLESS this is the last factor, in which case remaining/d can be anything > 12
        # But we require all factors > 12, so remaining/d > 12 if it's a factor
        
        # Let me just enumerate all valid d and recurse
        
        for d in all_divisors:
            if d < min_factor:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            
            # Check if new_remaining can be decomposed into distinct factors > 12, >= d
            # (to maintain distinctness and ordering)
            # Minimum possible: if new_remaining == 1, we're done (but d was the last factor)
            # Actually if new_remaining == 1, that means d == remaining, so we just add d
            
            if new_remaining == 1:
                # d is the last factor
                if d > 12:
                    new_max = max(current_max, d)
                    if new_max < best_max[0]:
                        best_max[0] = new_max
                        best_factors[0] = list(current_factors) + [d]
                continue
            
            # new_remaining > 1, need to factor it into factors >= d, > 12, distinct
            # Pruning: new_remaining must be >= d (since next factor >= d)
            # Actually next factor >= d, and if there are k more factors, min product is d*(d+1)*...
            # Simple check: new_remaining >= d (at least one more factor of size >= d)
            if new_remaining < d:
                continue
            
            # Additional pruning: the product of the smallest k distinct integers >= d
            # must be <= new_remaining for some k >= 1
            # For k=1: d <= new_remaining (already checked)
            # For k=2: d*(d+1) <= new_remaining
            # etc.
            # This is a necessary condition but not sufficient
            
            # Also, all factors must be distinct from current_factors
            # Since we're going in increasing order and current_factors are all < d (because min_factor = last factor + 1 or similar)
            # Actually min_factor ensures new factors are >= min_factor, and current_factors are all < min_factor
            # So distinctness from current_factors is automatic if we enforce >= min_factor
            
            # Recurse
            current_factors.append(d)
            dfs(new_remaining, d + 1, current_factors, max(current_max, d))
            current_factors.pop()
    
    # Start DFS: remaining = N, min_factor = 13 (smallest > 12), no factors yet
    # But we also want to allow the single-factor solution as baseline
    # The DFS will find multi-factor solutions
    
    # To speed up, let's try different strategies:
    # Strategy 1: DFS from smallest factors
    # Strategy 2: Try to find balanced factorizations
    
    # Let's also try a targeted approach: for k factors, try to make them balanced
    # k can range from 2 to 12
    
    # Actually, let's just run the DFS but with better pruning
    # Key insight: if current_max is already >= best_max[0], we can prune
    # because adding more factors won't reduce the max below current_max
    
    # Let me restructure with this pruning
    
    def dfs2(remaining, min_factor, current_factors, current_max):
        if time.time() > deadline:
            return
        
        # Pruning: if current_max >= best_max[0], no improvement possible
        if current_max >= best_max[0] and current_factors:
            # We could still get a solution with same max but... we want strictly better
            # Actually if current_max == best_max[0], we can't improve, so prune
            return
        
        if remaining == 1:
            if current_factors:
                m = max(current_factors)
                if m < best_max[0]:
                    best_max[0] = m
                    best_factors[0] = list(current_factors)
            return
        
        # We need at least one more factor
        # The factor must be >= min_factor, > 12, divide remaining
        # And remaining/factor must be decomposable into factors >= factor
        
        # If remaining < min_factor, impossible
        if remaining < min_factor:
            return
        
        # If remaining is prime (or has no divisors >= min_factor), we can only use it as a single factor
        # Check if remaining itself is a valid factor
        # Actually, let's just enumerate
        
        # Enumerate divisors of remaining that are >= min_factor
        # For efficiency, we could precompute divisors, but remaining changes
        # Let's compute divisors of remaining on the fly
        
        # Find all divisors of remaining
        divs = []
        i = 1
        while i * i <= remaining:
            if remaining % i == 0:
                if i >= min_factor:
                    divs.append(i)
                other = remaining // i
                if other >= min_factor and other != i:
                    divs.append(other)
            i += 1
        divs.sort()
        
        for d in divs:
            new_remaining = remaining // d
            
            if new_remaining == 1:
                if d > 12:
                    new_max = max(current_max, d)
                    if new_max < best_max[0]:
                        best_max[0] = new_max
                        best_factors[0] = list(current_factors) + [d]
                continue
            
            # new_remaining > 1
            # Need to factor new_remaining into factors >= d
            # Simple check: new_remaining >= d
            if new_remaining < d:
                continue
            
            # Additional pruning: new_remaining must have a divisor >= d
            # (which is true since d divides remaining, so new_remaining = remaining/d, 
            #  and we need factors of new_remaining >= d)
            # The largest factor of new_remaining is new_remaining itself
            # So we need new_remaining >= d (already checked)
            
            # Prune if current_max is already too high
            if current_factors and max(current_max, d) >= best_max[0]:
                continue
            
            current_factors.append(d)
            dfs2(new_remaining, d + 1, current_factors, max(current_max, d))
            current_factors.pop()
    
    # Run DFS
    dfs2(N, 13, [], 0)
    
    # Return the best solution found
    return best_factors[0]
# EVOLVE-BLOCK-END
