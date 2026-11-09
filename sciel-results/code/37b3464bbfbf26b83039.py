import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    target = Fraction(1, 1)
    min_denom = 5
    max_denom = 10000000
    
    def dfs(current_sum, last_denom, remaining, path):
        if time.time() > deadline:
            return None
            
        if remaining == 0:
            if current_sum == target:
                return path
            return None
            
        if current_sum > target:
            return None
            
        rem_val = target - current_sum
        if rem_val <= 0:
            return None
            
        # Lower bound for d
        # 1/d <= rem_val => d >= 1/rem_val
        min_d_frac = Fraction(1, 1) / rem_val
        min_d = min_d_frac.numerator // min_d_frac.denominator
        if min_d_frac.numerator % min_d_frac.denominator != 0:
            min_d += 1
        min_d = max(min_d, last_denom + 1)
        
        # Upper bound for d
        # Sum of remaining terms must be at least rem_val
        # If d is too large, 1/d is too small.
        # Max sum of 'remaining' terms is roughly remaining / d (if all were d, but they are distinct and >= d)
        # Actually, max sum is 1/d + 1/(d+1) + ...
        # If 1/d + (remaining-1)/1 < rem_val is not a good bound.
        # Use: if 1/d < rem_val / remaining, then even if we had 'remaining' terms of 1/d, sum < rem_val.
        # So d > remaining / rem_val is a valid upper bound to skip.
        max_d_frac = Fraction(remaining, 1) / rem_val
        max_d = max_d_frac.numerator // max_d_frac.denominator
        if max_d > max_denom:
            max_d = max_denom
            
        if min_d > max_d:
            return None
            
        for d in range(min_d, max_d + 1):
            new_sum = current_sum + Fraction(1, d)
            
            if new_sum > target:
                # Need larger d to reduce sum
                continue
                
            # Pruning: can we reach target with remaining-1 terms?
            # Max possible sum from next terms: 1/(d+1) + 1/(d+2) + ...
            # If new_sum + max_possible < target, break (larger d makes it worse)
            if remaining > 1:
                # Approximate max possible sum
                # Sum_{i=1}^{remaining-1} 1/(d+i)
                # This is < (remaining-1)/(d+1)
                # If new_sum + (remaining-1)/(d+1) < target, break
                if new_sum + Fraction(remaining - 1, d + 1) < target:
                    break
                    
            res = dfs(new_sum, d, remaining - 1, path + [d])
            if res is not None:
                return res
                
        return None

    # Search for minimal size
    for k in range(8, 12):  # Try 8, 9, 10, 11
        res = dfs(Fraction(0, 1), 4, k, [])
        if res is not None:
            return res
            
    # Fallback
    raise RuntimeError("No solution found")
