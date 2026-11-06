import math
import time

def construct():
    target = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute divisors of target to speed up iteration if needed, 
    # but integer iteration with pruning is likely sufficient.
    # Let's use integer iteration for simplicity and correctness.
    
    def dfs(rem_prod, min_val, k_left, current_factors):
        if k_left == 1:
            if rem_prod > min_val and rem_prod > 12:
                # Check if rem_prod is integer (it is) and valid
                # We also need to ensure it doesn't violate distinctness 
                # which is handled by min_val > last_factor
                return [rem_prod]
            return None
        
        # Max value for current factor x
        # x^k_left <= rem_prod  => x <= rem_prod^(1/k_left)
        # Also x must be <= M (handled by caller or passed in)
        # We pass M as a closure variable or argument.
        
        max_x = int(rem_prod ** (1.0 / k_left))
        if max_x < min_val:
            return None
            
        # Pruning lower bound:
        # The smallest product of k_left-1 distinct integers > x is (x+1)...(x+k_left-1)
        # If (x+1)...(x+k_left-1) > rem_prod // x, then no solution.
        
        for x in range(min_val, max_x + 1):
            if time.time() > deadline:
                return None # Timeout, return failure
            
            if rem_prod % x != 0:
                continue
                
            new_rem = rem_prod // x
            
            # Check if new_rem can be formed by k_left-1 distinct factors > x
            # Lower bound on product of next k_left-1 factors:
            # (x+1) * (x+2) * ... * (x + k_left - 1)
            lb_prod = 1
            for i in range(1, k_left):
                lb_prod *= (x + i)
                if lb_prod > new_rem:
                    break
                    
            if lb_prod > new_rem:
                # Since x is increasing, lb_prod will only increase
                break
                
            # Upper bound check?
            # Max product of k_left-1 factors <= M is M^(k_left-1)
            # If new_rem > M^(k_left-1), then no solution for this x.
            # But we are iterating M in the outer loop, so M is fixed.
            # We can add this check.
            
            res = dfs(new_rem, x + 1, k_left - 1, current_factors + [x])
            if res is not None:
                return res
        return None

    # Iterate M from lower bound upwards
    # Lower bound for M:
    # If k=1, M=12!
    # If k=12, min M is 24 (but 13..24 product > 12!, so k<12)
    # We can start M from 13.
    
    # To optimize, we can try larger k first for a given M?
    # No, we just need to find ANY solution for the current M.
    # We iterate k from 1 to 12.
    
    for M in range(13, 1000):
        if time.time() > deadline:
            break
            
        found = False
        best_factors = None
        
        # Try different numbers of factors k
        # k=1 is trivial: [12!] but M must be >= 12!, so this will be found last.
        # We want to minimize M, so we check all k for this M.
        for k in range(1, 13):
            # Quick check: can we even fit k factors in (12, M]?
            # Smallest k distinct integers > 12 are 13, 14, ..., 12+k
            # If 12+k > M, then impossible for this k.
            if 12 + k > M:
                continue
                
            res = dfs(target, 13, k, [])
            if res is not None:
                # Verify solution
                prod = 1
                for f in res:
                    prod *= f
                if prod == target and len(res) == k and all(f > 12 for f in res) and len(set(res)) == k:
                    best_factors = res
                    found = True
                    break # Found solution for this M, no need to check other k for this M? 
                    # Wait, we want to return the factors. Any valid set for min M is fine.
                    # The objective is to minimize the LARGEST factor. 
                    # The evaluator measures the raw value (largest factor).
                    # So we just need to return the set.
        
        if found:
            return best_factors

    # If no solution found within deadline or M range, return a valid fallback
    # A valid fallback is [12!] (1 factor, largest is 12!)
    # But we should try to find something better if possible.
    # The loop above should have found something.
    return [target]
