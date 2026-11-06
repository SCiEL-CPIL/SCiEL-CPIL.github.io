from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    target = Fraction(5, 31)
    best_max = 10000000
    best_solution = None
    
    # Search for solutions with k terms
    # k=1, 2 have no solutions.
    # We'll search k=3 to 8
    
    def dfs(remaining, min_d, current, k_remaining):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if len(current) > 0:
                curr_max = max(current)
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = current.copy()
            return
        
        if k_remaining == 0:
            return
        
        # Pruning:
        # Max possible sum with k_remaining terms starting from min_d
        # Approx 1/min_d + 1/(min_d+1) ...
        # If remaining > sum, fail
        # Min possible sum is 0.
        
        # Also, if remaining * min_d < 1, it might still be possible if min_d is small?
        # 1/d <= remaining => d >= 1/remaining.
        # So min_d must be <= 1/remaining? No, d >= ceil(1/remaining).
        
        # Lower bound for next denominator d:
        # 1/d <= remaining => d >= 1/remaining.
        # Also d > min_d.
        d_min = max(min_d + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound for d:
        # We need to be able to form the remaining sum with k_remaining - 1 terms.
        # The smallest possible sum with k_remaining - 1 terms is 0.
        # The largest possible sum with k_remaining - 1 terms starting from d+1 is ...
        # Actually, if 1/d >= remaining, we can't use d (unless k_remaining=1).
        # If k_remaining > 1, we need 1/d < remaining.
        # Also, the remaining sum (remaining - 1/d) must be representable by k_remaining - 1 terms.
        # The max sum of k_remaining - 1 terms starting from d+1 is approx (k_remaining - 1) / (d+1).
        # So remaining - 1/d <= (k_remaining - 1) / (d+1).
        
        # Let's just iterate d from d_min upwards.
        # How far?
        # If k_remaining == 1:
        #   1/d = remaining => d = 1/remaining. Must be integer.
        if k_remaining == 1:
            d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            if d >= d_min and 1/d == remaining:
                current.append(d)
                dfs(0, d, current, 0)
                current.pop()
            return
        
        # Estimate max d
        # remaining - 1/d > 0 => d > 1/remaining.
        # remaining - 1/d < (k_remaining - 1) / (d+1)
        # (remaining - 1/d)(d+1) < k_remaining - 1
        # remaining*d + remaining - 1 - 1/d < k_remaining - 1
        # remaining*d < k_remaining - 2 + 1 - remaining + 1/d ...
        
        # Simpler bound:
        # d < 1 / (remaining / k_remaining) = k_remaining / remaining.
        # So d_max = floor(k_remaining / remaining).
        
        d_max = (k_remaining * remaining.denominator) // remaining.numerator
        if d_max < d_min:
            return
            
        # To keep it fast, we can cap d_max if it's huge, but here numbers are small.
        
        for d in range(d_min, d_max + 1):
            if 1/d >= remaining:
                break
            current.append(d)
            dfs(remaining - 1/d, d, current, k_remaining - 1)
            current.pop()
            
            # Optional: if we found a solution with max < current best, we can prune more?
            # Not easily.
            
    # Run search for k=3 to 8
    for k in range(3, 9):
        dfs(target, 1, [], k)
        
    if best_solution:
        return best_solution
    else:
        # Fallback to known 3-term solution
        return [8, 31, 248]
