import time
import math
from math import gcd

def construct():
    # Known best solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Precompute harmonic numbers for pruning
    # H[n] = sum_{k=1}^{n} 1/k
    # We'll use floating point for speed in pruning checks
    max_harmonic_n = 20000  # Precompute up to a reasonable limit
    H = [0.0] * (max_harmonic_n + 1)
    for i in range(1, max_harmonic_n + 1):
        H[i] = H[i-1] + 1.0 / i
    
    def harmonic_sum(lo, hi):
        """Return H[hi] - H[lo-1] as float approximation of sum_{k=lo}^{hi} 1/k"""
        if lo > hi:
            return 0.0
        if hi > max_harmonic_n:
            # Approximate for large values
            return math.log(hi) - math.log(lo - 1) if lo > 1 else math.log(hi) + 0.5772
        return H[hi] - (H[lo - 1] if lo > 1 else 0.0)
    
    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        """
        
        def dfs(num, den, min_d, current_set):
            # Time check
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Pruning 1: Lower bound for next denominator d
            # 1/d <= num/den  => d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Pruning 2: Upper bound check - max possible sum from lower_bound to limit
            max_possible = harmonic_sum(lower_bound, limit)
            if max_possible < num / den - 1e-15:
                return None
            
            # Pruning 3: Check if we have enough terms available
            # Each term is at most 1/min_d, so we need at least ceil(num*min_d/den) terms
            min_terms_needed = (num * min_d + den - 1) // den
            available_terms = limit - lower_bound + 1
            if min_terms_needed > available_terms:
                return None
            
            # Pruning 4: Also check with the actual lower_bound
            min_terms_needed2 = (num * lower_bound + den - 1) // den
            if min_terms_needed2 > available_terms:
                return None
            
            # Iterate possible denominators
            # Try a mix: start from lower_bound but also consider larger values
            # For better pruning, try denominators that leave "nice" remainders
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Check if 1/d <= num/den (should be true by lower_bound, but be safe)
                if num * d < den:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Quick prune: check if remainder can be achieved
                if next_num > 0:
                    # Max sum from d+1 to limit
                    max_rem = harmonic_sum(d + 1, limit)
                    if max_rem < next_num / next_den - 1e-15:
                        continue
                    # Min terms check
                    min_t = (next_num * (d + 1) + next_den - 1) // next_den
                    if min_t > limit - d:
                        continue
                
                # Recurse
                res = dfs(next_num, next_den, d + 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
            
        # Start with remainder 5/31, min_denom 1
        return dfs(5, 31, 1, [])

    # Main Search Loop
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
            
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Now try to find a solution with even smaller max
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            break
            
    return best_solution
