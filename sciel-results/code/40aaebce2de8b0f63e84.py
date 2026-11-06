import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s margin out of 90s
    
    N = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # Primes: 2^10, 3^5, 5^2, 7^1, 11^1
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # We want to find distinct integers > 12 whose product is N.
    # Minimize the largest factor.
    
    # Let's precompute divisors of N that are > 12.
    # Actually, we can just use a recursive search with pruning.
    
    # Key insight: The factors must be distinct and > 12.
    # Since we want to minimize the max factor, we should try to keep factors balanced.
    # But they must be distinct.
    
    # Let's use a DFS that picks factors in increasing order.
    # State: remaining product, last chosen factor, list of chosen factors.
    
    # Pruning:
    # 1. If remaining == 1, we are done. Check if solution is valid (distinct, >12).
    # 2. If remaining is prime and > last_factor, we can add it as the last factor.
    # 3. If remaining is composite, we can try divisors.
    
    # To speed up, we can precompute the divisors of N.
    # But N is fixed, so we can precompute all divisors of N that are > 12.
    
    # Let's generate all divisors of N.
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Filter divisors > 12
    divisors_gt12 = [d for d in divisors if d > 12]
    
    # For fast lookup, we can use a set.
    div_set = set(divisors_gt12)
    
    # Also, we need to check if a number divides another.
    # We can just use modulo.
    
    best_max = N
    best_solution = [N]
    
    def dfs(remaining, last_factor, current_factors):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            if len(current_factors) >= 1:
                max_f = current_factors[-1]
                if max_f < best_max:
                    best_max = max_f
                    best_solution = current_factors[:]
                    return True
            return False
        
        # If remaining is prime, it must be the last factor.
        # But we need to check if it's > last_factor and > 12.
        # Also, we need to ensure we don't exceed 12 factors.
        
        # Try to add remaining as the last factor
        if len(current_factors) + 1 <= 12:
            if remaining > last_factor and remaining > 12:
                max_f = remaining
                if max_f < best_max:
                    best_max = max_f
                    best_solution = current_factors[:] + [remaining]
                    return True
        
        # Try to find divisors of remaining that are > last_factor and > 12
        # We iterate through divisors of N that divide remaining.
        # But that's O(num_divisors) per call.
        # Better: iterate through divisors_gt12 that are > last_factor and divide remaining.
        
        # To optimize, we can break early if the divisor is >= best_max.
        # Also, we can break if the divisor is > remaining.
        
        for d in divisors_gt12:
            if d <= last_factor:
                continue
            if d >= best_max:
                break # Since divisors_gt12 is sorted, all subsequent are >= best_max
            if remaining % d != 0:
                continue
            
            # d is a valid candidate
            # Check if we can form the rest
            new_remaining = remaining // d
            # The next factors must be > d.
            # Also, the product of the rest must be new_remaining.
            # And the rest must be distinct from current_factors and d.
            
            # Pruning: if new_remaining is 1, we are done (handled above)
            # If new_remaining is prime, it must be the next factor.
            # If new_remaining is composite, we recurse.
            
            # Check if new_remaining is > d (since next factor must be > d)
            # Wait, if new_remaining is 1, we are done.
            # If new_remaining > 1, the next factor must be > d.
            # So new_remaining must be > d? No, new_remaining is the product of the rest.
            # The next factor f must satisfy f > d and f divides new_remaining.
            # So new_remaining must have a divisor > d.
            # This implies new_remaining > d.
            
            if new_remaining <= d:
                # If new_remaining is 1, we are done (handled above)
                # If new_remaining > 1, we need a factor > d, but new_remaining <= d, so impossible.
                continue
            
            dfs(new_remaining, d, current_factors + [d])
            
            # If we found a solution, we can stop early?
            # No, we want to find the best max.
            # But if we found a solution with max_f < best_max, we update best_max.
            # Then subsequent branches with d >= best_max are pruned.
        
        return False
        
    dfs(N, 12, [])
    
    return best_solution
