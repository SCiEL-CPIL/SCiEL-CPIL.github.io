import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors of target
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute minimum product for k distinct integers starting from start_val
    # min_prod[k][start_val] is not feasible as start_val is large.
    # Instead, compute on the fly or use a helper.
    
    best_max_factor = target
    best_solution = [target]
    
    # Helper to check if remaining can be formed by m distinct factors > last
    # Smallest product is (last+1)*(last+2)*...*(last+m)
    def min_possible_product(last, m):
        prod = 1
        for i in range(1, m + 1):
            prod *= (last + i)
            if prod > target: # Overflow protection, though not strictly needed here
                return target + 1
        return prod

    def dfs(remaining, last_factor, current_factors, depth):
        nonlocal best_max_factor, best_solution
        
        if time.time() > deadline:
            return
            
        if remaining == 1:
            # Found a valid decomposition
            if len(current_factors) >= 1:
                max_f = max(current_factors)
                if max_f < best_max_factor:
                    best_max_factor = max_f
                    best_solution = current_factors[:]
            return

        # Pruning 1: If remaining is prime, it must be the next factor
        # (Simplified: we just check in the loop)
        
        # Determine max number of factors we can still add
        # We have 12 slots total. We have used 'depth'.
        # We can add at most 12 - depth factors.
        # Also, we must add at least 1 factor if remaining > 1.
        
        # Iterate through possible next factors
        # We only need to consider divisors of 'remaining'
        # that are > last_factor and < best_max_factor
        
        # Optimization: Instead of iterating all valid_divs, 
        # we can iterate divisors of 'remaining' specifically.
        # But valid_divs is sorted, so we can break early.
        
        # Feasibility Prune:
        # If we use 1 more factor (the last one), it must be 'remaining'.
        # If we use more, the minimum product of those factors must be <= remaining.
        
        # Let's try to find a solution.
        # To find the minimum max factor, we should explore smaller factors first?
        # No, if we pick a small factor, the remaining product is large, 
        # forcing a larger last factor.
        # If we pick a large factor (but < best_max), the remaining product is smaller.
        # However, the constraint is "distinct and > last".
        
        # Standard DFS with branch and bound works well here.
        
        # We iterate d in valid_divs
        # But we can restrict to divisors of 'remaining'
        # To do this efficiently without generating divisors of 'remaining' each time,
        # we just check `remaining % d == 0`.
        
        # We can also prune if d >= best_max_factor.
        
        # Also, we need to ensure that the remaining product after dividing by d
        # can be factored into distinct integers > d.
        
        # Let's add a check:
        # If remaining // d < d + 1, it's impossible to have a next factor > d 
        # unless remaining // d == 1.
        
        for d in valid_divs:
            if d <= last_factor:
                continue
            if d >= best_max_factor:
                break # Since valid_divs is sorted
            
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            rem2 = remaining // d
            
            # Check if rem2 can be formed by distinct factors > d
            # If rem2 == 1, we are done (handled in next recursion or here?)
            # If we recurse, remaining=1 will trigger the base case.
            
            # Prune: If rem2 > 1, we need at least one more factor > d.
            # So rem2 must have a divisor > d.
            # Smallest such divisor is d+1 (if it divides) or larger.
            # Simple check: rem2 must be > d.
            if rem2 > 1 and rem2 < d + 1:
                continue
                
            # Stronger Prune:
            # If we assume we need to fill the rest with the smallest possible distinct integers > d,
            # say we use k more factors.
            # The minimum product of k distinct integers > d is (d+1)...(d+k).
            # If rem2 < (d+1), impossible (handled above).
            # If rem2 < (d+1)(d+2), then we can't use 2 more factors.
            # We don't know k exactly, but k <= 12 - depth - 1.
            
            # Let's just recurse. The pruning `d < best_max_factor` is the main driver.
            
            current_factors.append(d)
            dfs(rem2, d, current_factors, depth + 1)
            current_factors.pop()
            
            # If we found a better solution, best_max_factor is updated.
            # The loop will break earlier in the next iteration or current iteration if d >= new best.
            # But since we are in the loop, we just continue to next d.
            # If d >= best_max_factor, the loop breaks at the start of the next iteration.

    dfs(target, 12, [], 0)
    
    if best_solution is None or best_solution == [target]:
        return [target]
        
    return best_solution
