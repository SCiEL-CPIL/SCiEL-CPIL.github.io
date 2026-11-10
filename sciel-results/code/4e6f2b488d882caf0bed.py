import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter divisors > 12 and < target (proper divisors are candidates for intermediate steps, 
    # but the last factor can be target if it's the only one, but we start with 12!)
    # Actually, we just need divisors > 12.
    # We sort them to iterate in increasing order.
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    best_max_factor = target
    best_solution = [target]
    
    # Helper to check if a number is prime (for quick prune)
    # Since max divisor is ~4.8e8, simple trial division up to sqrt is ~22000, which is fast enough
    # if done rarely. But we can just rely on the divisor list.
    
    def dfs(remaining, last_factor, current_factors, depth):
        nonlocal best_max_factor, best_solution
        
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            if len(current_factors) >= 1:
                # Found a valid decomposition
                # The factors are in current_factors
                # Since we iterate in increasing order and enforce d > last_factor, 
                # the last element is the largest.
                max_f = current_factors[-1]
                if max_f < best_max_factor:
                    best_max_factor = max_f
                    best_solution = list(current_factors)
            return True
        
        if len(current_factors) >= 12:
            return False
            
        # Pruning:
        # The next factor d must be > last_factor.
        # Also, d must divide remaining.
        # If remaining is prime, it must be the last factor.
        # We can check this quickly.
        # But iterating divisors is fine.
        
        # Find start index for binary search or linear scan
        # Since valid_divs is sorted, we can find the first d > last_factor
        # Using linear scan from a cached index is hard in recursion without passing index.
        # Binary search is O(log N). N ~ 700. log2(700) ~ 10. Very fast.
        
        # We can also prune if remaining < last_factor + 1
        if remaining < last_factor + 1:
            return False
            
        # Iterate through valid divisors
        # To avoid re-scanning from 0, we can pass a start index, 
        # but binary search is simple enough.
        # Let's use a simple loop with break conditions.
        
        # Optimization: Pre-compute divisors of 'remaining'?
        # No, 'remaining' changes.
        
        # We iterate d in valid_divs
        # If d <= last_factor, skip
        # If d > remaining, break
        # If remaining % d != 0, skip
        # If d >= best_max_factor, break (since we want strictly smaller max, 
        # and d is the current max if it's the last, or less than future max)
        # Actually, if d >= best_max_factor, any solution extending this will have max >= d >= best_max_factor.
        # So we can break.
        
        for d in valid_divs:
            if d <= last_factor:
                continue
            if d > remaining:
                break
            if d >= best_max_factor:
                break
                
            if remaining % d != 0:
                continue
                
            rem2 = remaining // d
            
            # Additional prune:
            # If rem2 > 1, we need at least one more factor > d.
            # The smallest such factor is d+1.
            # So rem2 must be >= d+1.
            if rem2 > 1 and rem2 < d + 1:
                continue
                
            # Recursive step
            current_factors.append(d)
            dfs(rem2, d, current_factors, depth + 1)
            current_factors.pop()
            
            # If we found a better solution, best_max_factor is updated.
            # The loop will break if next d >= new best_max_factor.
            
        return False

    dfs(target, 12, [], 0)
    
    if best_solution is None or best_solution == [target]:
        return [target]
        
    return best_solution
