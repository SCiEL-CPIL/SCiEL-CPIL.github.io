import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of TARGET
    # We can generate them by iterating through exponents of prime factors
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # Map divisor to index for quick lookup? Or just use set for membership?
    # Set is fine.
    valid_div_set = set(valid_divisors)
    
    best_max_factor = TARGET # Initial best is the single factor
    best_solution = [TARGET]
    
    def dfs(remaining, last_factor, current_factors, depth):
        nonlocal best_max_factor, best_solution
        
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            if len(current_factors) >= 1:
                # Found a valid solution
                max_f = max(current_factors)
                if max_f < best_max_factor:
                    best_max_factor = max_f
                    best_solution = list(current_factors)
            return True
        
        # Pruning:
        # 1. If remaining is prime and > last_factor, it must be the last factor.
        #    But it must be distinct and > 12.
        # 2. The next factor d must be a divisor of remaining.
        # 3. d > last_factor.
        # 4. d > 12.
        # 5. d must be such that remaining/d can be factored into distinct factors > d.
        
        # Lower bound for next factor:
        # The minimal product of (max_depth - depth) distinct factors > last_factor
        # is not directly useful for the *next* factor, but we can check if remaining
        # can be formed by factors all > last_factor.
        
        # A simple strong prune:
        # If remaining is prime, it must be the next factor.
        # Check if remaining > last_factor and remaining > 12.
        
        # Iterate through possible next factors
        # We can iterate through all divisors of 'remaining' that are > last_factor and > 12
        # To do this efficiently, we can iterate through valid_divisors and check if they divide remaining.
        # However, valid_divisors is sorted. We can find the start index using binary search or just iterate.
        # Since valid_divisors size is ~700, iterating is fine.
        
        # Optimization: 
        # The next factor d must satisfy: d > last_factor
        # Also, the remaining product rem2 = remaining // d
        # rem2 must be factored into distinct factors all > d.
        # This implies rem2 >= (d+1)(d+2)...(d + k_remaining)
        # Where k_remaining is the number of factors left.
        # We don't know k_remaining exactly, but we know it's at least 1 (unless rem2=1) and at most (12 - depth).
        
        # Let's just iterate.
        for d in valid_divisors:
            if d <= last_factor:
                continue
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            rem2 = remaining // d
            
            # Pruning check:
            # Can rem2 be factored into distinct integers > d?
            # If rem2 == 1, we are done with this branch (handled above if we check after adding d? No, we add d then recurse).
            # If rem2 > 1, we need at least one more factor > d.
            # So rem2 must have a divisor > d.
            # Also, the product of the next (k-1) factors must be rem2.
            # The smallest product of (k-1) distinct integers > d is (d+1)(d+2)...
            # Let's do a quick check: if rem2 < d + 1, it's impossible.
            if rem2 > 1 and rem2 < d + 1:
                continue
                
            # Recursive step
            current_factors.append(d)
            found = dfs(rem2, d, current_factors, depth + 1)
            current_factors.pop()
            
            if found:
                # If we found a better solution, we might continue to see if we can do even better?
                # The problem asks to minimize the LARGEST factor.
                # Since we are iterating d in increasing order, and we enforce d > last_factor,
                # the first solution found with a specific 'max' might not be the global minimum max.
                # However, branch and bound: if we find a solution with max M, we can prune any branch
                # where the current factor d >= M (since d will be part of the max or the max will be larger).
                # Actually, if we find a solution, we update best_max_factor.
                # We should continue searching to see if we can find a solution with a smaller max.
                # But we can prune branches where d >= best_max_factor.
                pass
            
            # Prune if current d is already >= best_max_factor
            # If d >= best_max_factor, any solution extending this will have max >= d >= best_max_factor.
            # So we can break if d >= best_max_factor?
            # Yes, because valid_divisors is sorted. All subsequent d' > d will also be >= best_max_factor.
            if d >= best_max_factor:
                break

    dfs(TARGET, 12, [], 0)
    
    # Fallback if no solution found (should not happen)
    if best_solution is None or best_solution == [TARGET]:
        return [TARGET]
        
    return best_solution
