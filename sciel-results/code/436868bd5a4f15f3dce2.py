import math

def construct():
    target = math.factorial(12)
    
    # We want to find the minimum M such that target can be decomposed 
    # into distinct integers > 12 and <= M.
    # We iterate M upwards from 13.
    
    def can_form(rem, max_factor):
        # rem: remaining product to achieve
        # max_factor: maximum allowed value for the next factor (to ensure distinctness and order)
        if rem == 1:
            return True
        if rem < 13:
            return False
        # The next factor d must be <= max_factor and >= 13
        # Also d must divide rem
        # To find a solution quickly, we can iterate d downwards from max_factor to 13
        # However, we can prune: if rem < 13, fail.
        # Also, if we pick d, the new rem is rem//d.
        # The next factors must be < d.
        
        # Optimization: 
        # If we have k factors left to pick (unknown), the smallest product of k distinct factors < d is (d-1)*(d-2)*...
        # This is complex. Simple iteration is likely fast enough given small M.
        
        # Iterate d from min(max_factor, rem) down to 13
        # Note: if d > rem, we can't pick it (unless rem==d, then next rem is 1)
        # Actually if d == rem, we pick it and rem becomes 1.
        
        # We should check divisors.
        # To be efficient, we can just check d downwards.
        # Since M is expected to be small (< 100), this is fast.
        
        # Pruning: 
        # If we pick d, the remaining product is rem//d.
        # The minimum product of (number of remaining factors) distinct integers < d is hard to estimate without knowing count.
        # But we can use: if rem//d < 13 and rem//d != 1, then we need at least one more factor.
        # If rem//d is prime and > d-1, we can't form it with smaller factors?
        
        for d in range(min(max_factor, rem), 12, -1):
            if rem % d == 0:
                if can_form(rem // d, d - 1):
                    return True
        return False

    for M in range(13, target + 1):
        if can_form(target, M):
            # We found a valid M. We need to return the factors.
            # The previous function only returned True/False.
            # We need to modify it to return the factors or store them.
            pass
    
    # Re-implement to return the actual factors
    def find_factors(rem, max_factor):
        if rem == 1:
            return []
        if rem < 13:
            return None
        
        for d in range(min(max_factor, rem), 12, -1):
            if rem % d == 0:
                res = find_factors(rem // d, d - 1)
                if res is not None:
                    return [d] + res
        return None

    for M in range(13, target + 1):
        # Quick check first? Or just call find_factors
        # find_factors will find *a* solution if one exists.
        # Since we iterate M upwards, the first M that yields a solution is the optimal one.
        # However, find_factors might be slow if it explores many branches.
        # But the tree is small.
        
        # To ensure we don't timeout, we can add a quick check.
        # But given the small depth (max 7-8 factors), it should be fine.
        
        factors = find_factors(target, M)
        if factors is not None:
            return factors
            
    # Fallback (should not reach here)
    return [target]
