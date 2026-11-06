"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import factorial, gcd
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    MIN_VAL = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    num_primes = len(primes)
    
    # Generate all divisors of N with their prime exponent vectors
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2**e2) * (3**e3) * (5**e5) * (7**e7) * (11**e11)
                        if val > MIN_VAL:
                            divisors.append((val, (e2, e3, e5, e7, e11)))
    
    divisors.sort()
    divisor_values = [d[0] for d in divisors]
    divisor_exps = [d[1] for d in divisors]
    n_div = len(divisors)
    
    # Build index: value -> list of indices (should be unique since values are unique)
    val_to_idx = {}
    for i, v in enumerate(divisor_values):
        val_to_idx[v] = i
    
    best_solution = None
    best_max = N  # worst case: single factor [N]
    
    def time_exceeded():
        return time.time() > deadline
    
    def dfs(remaining_exps, min_val, current_factors, current_max, num_factors_needed):
        nonlocal best_solution, best_max
        
        if time_exceeded():
            return
        
        if num_factors_needed == 0:
            if all(e == 0 for e in remaining_exps):
                if current_max < best_max:
                    best_max = current_max
                    best_solution = current_factors[:]
            return
        
        if any(e < 0 for e in remaining_exps):
            return
        
        # Pruning: if current_max >= best_max, we can't improve
        # (we want strictly less than best_max)
        if current_max >= best_max:
            return
        
        # Lower bound on the max factor we'll need:
        # The remaining product must be split into num_factors_needed factors
        # each >= min_val and distinct
        # The minimum possible max is at least the geometric mean
        remaining_product = 1
        for i, e in enumerate(remaining_exps):
            remaining_product *= primes[i] ** e
        
        if remaining_product == 1:
            if num_factors_needed == 0:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = current_factors[:]
            return
        
        # Geometric mean lower bound
        geo_mean = remaining_product ** (1.0 / num_factors_needed)
        # Each factor must be at least min_val, so max factor >= geo_mean
        # Also max factor >= min_val
        lower_bound_max = max(min_val, int(geo_mean) + 1)
        
        # If even the lower bound is >= best_max, no improvement possible
        if lower_bound_max >= best_max:
            return
        
        # Upper bound on the smallest factor we can pick:
        # If we pick factor f, the remaining (num_factors_needed-1) factors must
        # each be > f, so their product >= (f+1)(f+2)...(f+num_factors_needed-2)
        # And f * (f+1) * ... * (f+num_factors_needed-2) <= remaining_product
        # So f <= remaining_product^(1/num_factors_needed) approximately
        # More precisely: f^(num_factors_needed) <= remaining_product
        max_f = int(remaining_product ** (1.0 / num_factors_needed)) + 2
        
        # Also, the smallest factor must be >= min_val
        # And the largest factor must be < best_max (to improve)
        
        # Forward checking: for each candidate factor, check if remaining
        # exponents can be achieved
        for i in range(n_div):
            if time_exceeded():
                return
            
            f_val = divisor_values[i]
            
            if f_val < min_val:
                continue
            
            if f_val > max_f:
                break
            
            if f_val >= best_max:
                break
            
            # Check if f_val divides the remaining product (exponent check)
            f_exps = divisor_exps[i]
            can_use = True
            for j in range(num_primes):
                if f_exps[j] > remaining_exps[j]:
                    can_use = False
                    break
            if not can_use:
                continue
            
            # Forward checking: after using f, can we still factor the rest?
            new_remaining = [remaining_exps[j] - f_exps[j] for j in range(num_primes)]
            new_product = 1
            for j in range(num_primes):
                if new_remaining[j] > 0:
                    new_product *= primes[j] ** new_remaining[j]
            
            if num_factors_needed - 1 > 0:
                # The remaining (num_factors_needed-1) factors must each be > f_val
                # and distinct, with product = new_product
                # Minimum product of (num_factors_needed-1) distinct integers > f_val
                # is (f_val+1)(f_val+2)...(f_val+num_factors_needed-2)
                if num_factors_needed - 1 >= 1:
                    min_prod = 1
                    for j in range(1, num_factors_needed):
                        min_prod *= (f_val + j)
                        if min_prod > new_product:
                            break
                    if min_prod > new_product:
                        # Since f_val is increasing in our loop, this will only get worse
                        break
                
                # Also, the max of remaining factors must be < best_max
                # So new_product must be factorable into (num_factors_needed-1) factors
                # each < best_max and > f_val
                # Upper bound: (best_max-1)^(num_factors_needed-1) >= new_product
                if new_product > 0 and (best_max - 1) ** (num_factors_needed - 1) < new_product:
                    # Can't fit all remaining factors below best_max
                    # But this bound is loose; skip for now
                    pass
            
            new_max = max(current_max, f_val)
            dfs(new_remaining, f_val + 1, current_factors + [f_val], new_max, num_factors_needed - 1)
    
    # Try different numbers of factors
    # More factors generally allow smaller max
    # But minimum product of k distinct integers > 12 is 13*14*...*(13+k-1)
    for k in range(MAX_FACTORS, 1, -1):
        if time_exceeded():
            break
        
        # Check if k factors is feasible: min product of k distinct ints > 12
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
            if min_possible > N:
                break
        
        if min_possible > N:
            continue
        
        # Theoretical lower bound on max factor for k factors
        lower_bound = int(N ** (1.0 / k)) + 1
        
        # If we already have a solution with max <= lower_bound, can't improve
        if best_solution is not None and best_max <= lower_bound:
            break
        
        # Start DFS with k factors
        dfs(target_exps[:], MIN_VAL, [], 0, k)
        
        # Early termination if we found a very good solution
        if best_solution and best_max <= lower_bound + 1:
            break
    
    if best_solution is None:
        return [N]
    
    return best_solution

# EVOLVE-BLOCK-END
