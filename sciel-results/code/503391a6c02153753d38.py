"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import factorial
from functools import lru_cache

def construct():
    target = factorial(12)
    
    # 1. Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    
    divisors = sorted(divisors)
    
    # Precompute divisors for each divisor of target
    # divisors_of[x] = list of divisors of x that are also divisors of target
    # Since x is a divisor of target, its divisors are a subset of divisors.
    divisors_map = {}
    for d in divisors:
        divs = []
        for d2 in divisors:
            if d % d2 == 0:
                divs.append(d2)
        divisors_map[d] = divs

    best_max = float('inf')
    best_factors = []
    
    # DFS to find the set of distinct factors > 12 with product target
    # We enforce factors to be picked in strictly increasing order to avoid permutations
    # and ensure distinctness.
    
    def dfs(remaining, min_factor, current_factors, current_max):
        nonlocal best_max, best_factors
        
        if remaining == 1:
            if len(current_factors) > 0 and current_max < best_max:
                best_max = current_max
                best_factors = list(current_factors)
            return
        
        # Pruning: if current_max is already >= best_max, no point continuing
        if current_max >= best_max:
            return
            
        # Get divisors of remaining
        # We need to pick a divisor d of remaining such that d >= min_factor and d > 12
        # and d <= remaining (obviously)
        # Also, to minimize the max factor, we should try smaller d first?
        # If we pick a small d, we might be forced to pick a larger one later.
        # If we pick a larger d, we might reduce the number of factors needed.
        # However, the constraint is on the MAX factor.
        # So we want all factors to be small.
        # This suggests trying smaller factors first might lead to a solution with a smaller max,
        # but it's not guaranteed.
        # However, if we find ANY solution with max M, we can prune any branch where a factor > M is picked.
        
        # Optimization: 
        # The smallest possible factor we can pick is min_factor.
        # If min_factor > remaining, we can't pick any factor. Fail.
        if min_factor > remaining:
            return
            
        # Iterate through divisors of remaining
        # We only care about divisors d where d >= min_factor and d > 12
        # Also d <= remaining
        # To ensure we don't pick the same factor twice, we enforce d > previous factor (min_factor)
        
        # We can iterate through the precomputed list of divisors of 'remaining'
        # and filter
        for d in divisors_map[remaining]:
            if d < min_factor:
                continue
            if d <= 12:
                continue
            if d > remaining:
                break # divisors are sorted ascending? No, divisors_map[remaining] is sorted because 'divisors' was sorted.
            
            # Additional pruning:
            # If we pick d, the new remaining is remaining // d.
            # The next factor must be > d.
            # So the next factor must be at least d+1.
            # If remaining // d is prime, the next factor must be remaining // d.
            # So if remaining // d is prime and (remaining // d) <= d, we can't form a valid sequence (since next must be > d).
            # Actually, if remaining // d == 1, we are done.
            # If remaining // d > 1, we need to form it with factors > d.
            # The smallest factor > d is d+1.
            # If remaining // d < d+1, it's impossible unless remaining // d == 1.
            # So if remaining // d > 1 and remaining // d <= d:
            #   This implies the next factor must be exactly remaining // d (if it's prime) or a product.
            #   But all factors must be > d.
            #   So if remaining // d <= d, the only way is if remaining // d == 1.
            #   Wait, if remaining // d = 1, we are done.
            #   If remaining // d > 1, we need at least one factor.
            #   The smallest factor is > d.
            #   So the product of remaining factors is at least (d+1).
            #   So if remaining // d < d+1, it's impossible.
            #   i.e., if remaining // d <= d, it's impossible (unless remaining // d == 1).
            
            rem2 = remaining // d
            if rem2 > 1 and rem2 <= d:
                continue

            # Also, if rem2 is prime, it must be the next factor.
            # So if rem2 is prime and rem2 <= d, impossible (handled above).
            # If rem2 is prime and rem2 > d, we must pick rem2.
            # We can force this choice to reduce branching.
            
            current_factors.append(d)
            new_max = max(current_max, d)
            dfs(rem2, d + 1, current_factors, new_max)
            current_factors.pop()
            
            # If we found a solution, best_max is updated.
            # We can break if d >= best_max?
            # If d >= best_max, any solution from this branch will have max >= d >= best_max.
            # So we can break the loop if d >= best_max.
            if d >= best_max:
                break

    # Initial call
    # min_factor must be > 12. So min_factor = 13.
    # However, the smallest divisor of 12! greater than 12 is 16? No, 12! is divisible by 13? No.
    # 12! is not divisible by 13.
    # The smallest divisor > 12 is 16? 
    # Divisors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16...
    # 14 is a divisor. 15 is a divisor.
    # So min_factor = 13 is fine, the loop will skip 13 if it's not a divisor.
    
    dfs(target, 13, [], 0)
    
    if not best_factors:
        # Fallback: return a valid decomposition even if not optimal
        # e.g. [14, 15, 16, ...]
        # Or just return [target] if no solution found (should not happen)
        return [target]
        
    return best_factors

# EVOLVE-BLOCK-END
