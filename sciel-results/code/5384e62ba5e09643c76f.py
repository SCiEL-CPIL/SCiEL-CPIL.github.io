"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We'll use a DFS approach to build distinct factors > 12
    # that multiply to target, minimizing the largest factor.
    
    best_result = [target]  # fallback: single factor
    best_max = target
    
    # Precompute all divisors of target that are > 12
    # Actually, let's do a smarter search
    
    # Strategy: Try to find a set of distinct integers > 12 whose product is target
    # and whose maximum is minimized.
    # 
    # We'll use iterative deepening on the maximum factor value.
    # Start with a reasonable lower bound and increase.
    
    # Lower bound: if we use k factors, each must be > 12, distinct.
    # Minimum possible product with k distinct factors > 12: 13*14*...*12+k
    # We need this to be <= target.
    
    # Let's try different numbers of factors and search for valid decompositions.
    
    # For a given max value M, can we find distinct factors in (12, M] whose product is target?
    # We'll binary search M or just iterate.
    
    # First, find a feasible M. The theoretical minimum for largest factor:
    # With 12 factors: min product = 13*14*...*24
    # 13*14*15*16*17*18*19*20*21*22*23*24
    # Let's compute this
    
    def min_product_k_factors(k):
        """Minimum product of k distinct integers all > 12"""
        prod = 1
        for i in range(13, 13 + k):
            prod *= i
        return prod
    
    # Find minimum k such that min_product_k_factors(k) <= target
    min_k = 1
    for k in range(1, 13):
        if min_product_k_factors(k) <= target:
            min_k = k
            break
    
    # Now search for solutions with various numbers of factors
    # Use DFS with pruning
    
    def dfs(remaining_target, used_factors, current_factors, current_max, depth, max_depth):
        """
        remaining_target: the product we still need to achieve
        used_factors: set of factors already used
        current_factors: list of factors chosen so far
        current_max: max factor chosen so far
        depth: current depth
        max_depth: maximum number of factors allowed
        """
        nonlocal best_result, best_max
        
        if time.time() > deadline:
            return
        
        if remaining_target == 1:
            # We've found a complete factorization
            if current_factors:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best_result = list(current_factors)
            return
        
        if depth >= max_depth:
            return
        
        # Pruning: if current_max already >= best_max, no point continuing
        if current_max >= best_max and current_factors:
            return
        
        # Find possible next factors: divisors of remaining_target that are > 12
        # and not in used_factors, and >= current_max (to avoid permutations)
        # Actually, we should allow factors in any order but enforce distinctness.
        # To avoid permutations, let's enforce that next factor >= last chosen factor.
        
        if current_factors:
            min_next = current_factors[-1] + 1  # must be > last for distinctness and ordering
        else:
            min_next = 13
        
        # Find divisors of remaining_target that are >= min_next
        # and not in used_factors
        
        # Iterate through possible divisors
        # remaining_target can be large, so let's be smart
        # We only need divisors d where d >= min_next and d <= remaining_target
        
        # Find all divisors of remaining_target
        # But that could be expensive. Let's iterate up to sqrt(remaining_target)
        
        found_divisors = []
        if remaining_target > 1:
            # Find divisors
            i = 1
            # We need divisors >= min_next
            # Start from min_next, but that could be large
            # Better: find all divisors, then filter
            # For efficiency, iterate up to sqrt
            limit = int(remaining_target ** 0.5) + 1
            for i in range(1, limit):
                if remaining_target % i == 0:
                    if i >= min_next:
                        found_divisors.append(i)
                    other = remaining_target // i
                    if other != i and other >= min_next:
                        found_divisors.append(other)
        
        # Sort divisors to try smaller ones first (to keep max small)
        found_divisors.sort()
        
        for d in found_divisors:
            if d in used_factors:
                continue
            if d <= 12:
                continue
            
            new_max = max(current_max, d) if current_factors else d
            if current_factors and new_max >= best_max:
                continue
            
            # Additional pruning: the remaining product after dividing by d
            # must be achievable with (max_depth - depth - 1) more factors
            # each > d (since we're enforcing ordering) and not in used_factors
            rem = remaining_target // d
            
            if rem == 1:
                # This is the last factor
                if len(current_factors) + 1 <= max_depth:
                    new_factors = current_factors + [d]
                    mx = max(new_factors)
                    if mx < best_max:
                        best_max = mx
                        best_result = list(new_factors)
                continue
            
            # Check if rem can be factored into (max_depth - depth - 1) distinct factors
            # each > d and not in used_factors
            remaining_slots = max_depth - depth - 1
            if remaining_slots <= 0:
                continue
            
            # The minimum possible product for remaining_slots distinct factors each > d
            min_rem_product = 1
            for i in range(d + 1, d + 1 + remaining_slots):
                min_rem_product *= i
            
            if min_rem_product > rem:
                continue
            
            # Also, rem must be >= the minimum factor we could use
            # And rem must have enough "room"
            
            dfs(rem, used_factors | {d}, current_factors + [d], new_max, depth + 1, max_depth)
    
    # Try different numbers of factors from 1 to 12
    # Start with fewer factors (likely larger max) and work up
    # Actually, more factors likely means smaller max, so try more factors first
    
    for max_depth in range(12, 0, -1):
        if time.time() > deadline:
            break
        dfs(target, set(), [], 0, 0, max_depth)
        if best_max < target:
            break  # Found a non-trivial solution
    
    return best_result
# EVOLVE-BLOCK-END
