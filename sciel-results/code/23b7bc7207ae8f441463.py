import math
import time
import bisect

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # We need to find k distinct integers > 12 such that their product is N.
    # Minimize the largest factor.
    
    # 1. Prime factorization of N
    # 12! = 479001600
    # Primes <= 12: 2, 3, 5, 7, 11
    # 2^10 * 3^5 * 5^2 * 7 * 11
    prime_powers = [
        (2, 10),
        (3, 5),
        (5, 2),
        (7, 1),
        (11, 1)
    ]
    
    # Generate all divisors of N
    divisors = [1]
    for p, e in prime_powers:
        new_divs = []
        p_power = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute a set for O(1) lookup
    valid_divs_set = set(valid_divs)
    
    best_max_val = N
    best_factors = [N]
    
    def check_time():
        return time.time() - start_time > 55
    
    # Search for k from 7 down to 2.
    # k=1 is [N], which is the initial best.
    # k > 7 is impossible because 13*14*15*16*17*18*19*20 > N.
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        if check_time():
            return
            
        if depth == target_k:
            if current_prod == N:
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        target_rem = N // current_prod
        
        # Pruning: Minimum product of remaining distinct factors > last_factor
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
            
        if target_rem < min_prod:
            return
        
        # Pruning: If the minimum possible max factor for the remaining part
        # is already >= best_max_val, we can't improve.
        # The max factor of the remaining part is at least (target_rem)^(1/remaining).
        # Note: target_rem is the product of the remaining factors.
        min_possible_max_rem = target_rem ** (1.0 / remaining)
        if min_possible_max_rem >= best_max_val:
            return
        
        # If remaining == 1, the factor is fixed
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12 and x in valid_divs_set:
                # Check if this improves best
                if x < best_max_val:
                    path.append(x)
                    best_max_val = x
                    best_factors = list(path)
                    path.pop()
            return
        
        # Determine range for next factor x
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound estimation:
        # x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem  => x <= target_rem^(1/remaining)
        # Add a small margin to be safe
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        
        # Use bisect to find indices in valid_divs
        start_idx = bisect.bisect_left(valid_divs, lower_bound)
        end_idx = bisect.bisect_right(valid_divs, x_upper)
        
        # Iterate through valid divisors in range
        # Try larger candidates first to minimize the max factor faster?
        # If we pick a larger x, the remaining product is smaller, so remaining factors are smaller.
        # This should lead to smaller max factors.
        for i in range(end_idx - 1, start_idx - 1, -1):
            x = valid_divs[i]
            
            if target_rem % x != 0:
                continue
            
            # Additional pruning: x itself must be < best_max_val
            if x >= best_max_val:
                # Since we are iterating downwards, all subsequent x will be smaller.
                # But wait, if x >= best_max_val, we can't improve with this x.
                # However, smaller x might allow smaller max for the rest?
                # No, if x is large, the rest is small.
                # If x is small, the rest is large.
                # We want the MAX of all to be small.
                # If we pick a small x, the remaining product is large, so remaining factors are large.
                # So picking larger x is better for minimizing the max.
                # So if x >= best_max_val, we can't use it.
                # But we are iterating downwards. So we should break?
                # No, smaller x are < best_max_val.
                # But smaller x leads to larger remaining factors.
                # So it's not clear if we should break.
                # Just skip this x.
                continue

            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()
            
            if check_time():
                return

    # Search k=7 down to 2
    for k in range(7, 1, -1):
        # Check if it's worth searching k
        # N^(1/k) is the geometric mean.
        # If the geometric mean is already >= best_max_val, we can't do better.
        if N ** (1.0 / k) >= best_max_val:
            continue
            
        dfs(k, 1, 12, 0, [])
        
        if check_time():
            break

    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
