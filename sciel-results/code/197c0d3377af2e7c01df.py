import math
import time
import bisect

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute divisors of N
    divisors = set()
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    
    # Filter divisors > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    best_factors = [N]
    best_max_val = N
    
    # Helper to calculate product of range [start, start+len-1]
    # Optimized for small lengths
    def prod_range(start, length):
        p = 1
        for i in range(length):
            p *= (start + i)
        return p

    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        # Time check: keep a safety margin (90s limit, use 80s)
        if time.time() - start_time > 80:
            return
        
        if depth == target_k:
            if current_prod == N:
                # The factors are in path
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        
        if N % current_prod != 0:
            return
        target_rem = N // current_prod
        
        # Minimum possible product for the remaining 'remaining' factors
        # They must be distinct and > last_factor
        # So they are at least last_factor+1, last_factor+2, ...
        min_prod = prod_range(last_factor + 1, remaining)
        
        if target_rem < min_prod:
            return
        
        # If remaining == 1, the next factor must be target_rem
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12 and x in valid_divisors:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()
            return
        
        # Find candidates for the next factor x
        # x must be a divisor of target_rem
        # x > last_factor
        # x must be small enough such that the remaining factors (which are > x) can multiply to target_rem/x
        # Condition: target_rem / x >= prod_range(x + 1, remaining - 1)
        # => x * prod_range(x + 1, remaining - 1) <= target_rem
        # => prod_range(x, remaining) <= target_rem
        
        # We iterate through valid_divisors in ascending order
        # We can stop if prod_range(x, remaining) > target_rem
        
        # Find start index in valid_divisors
        left_idx = bisect.bisect_left(valid_divisors, last_factor + 1)
        
        for idx in range(left_idx, len(valid_divisors)):
            if time.time() - start_time > 80:
                return
            
            x = valid_divisors[idx]
            
            # Check if x divides target_rem
            if target_rem % x != 0:
                continue
            
            # Check upper bound condition
            # If x * (x+1) * ... * (x+remaining-1) > target_rem, 
            # then even the smallest possible remaining factors are too large.
            # We can break here if we assume valid_divisors is sorted and 
            # the product increases with x.
            min_possible_product_with_x = prod_range(x, remaining)
            if min_possible_product_with_x > target_rem:
                break
            
            # If we pass the check, recurse
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()

    # Try all k from 1 to 12
    # Note: k=1 is always valid (factor is 12!), max is 12!
    # We want to minimize the max, so we check all k.
    # However, larger k generally allows smaller max factors, but distinctness constraint kicks in.
    # 13*14*...*19 = 253955520 < 479001600.
    # 13*14*...*20 = 5079110400 > 479001600.
    # So max k is 7.
    
    # We should probably try k values that are likely to yield small max factors first?
    # Or just iterate all. The DFS will prune quickly.
    # Let's iterate k from 1 to 7.
    for k in range(1, 8):
        dfs(k, 1, 12, 0, [])
        
    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
