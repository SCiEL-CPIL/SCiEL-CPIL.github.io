"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin
    
    # Search for minimal k
    for k in range(8, 16):  # Try up to 15 terms
        if time.time() > deadline - 10:
            break
        result = _search(k, deadline)
        if result:
            return result
    # Fallback to seed-like behavior if search fails (should not happen)
    return _fallback()

def _search(k, deadline):
    # d[0] is the first denominator
    d = []
    
    def dfs(idx, current_sum, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if current_sum == 1:
                return d[:]
            return None
        
        # Pruning: Max possible sum for remaining terms
        # Remaining terms: k - idx
        # Smallest possible denominators: last_denom + 1, last_denom + 2, ...
        max_possible = 0
        for i in range(k - idx):
            max_possible += Fraction(1, last_denom + 1 + i)
        
        if current_sum + max_possible < 1:
            return None
        
        # Min possible sum is effectively 0, but we can check if current_sum > 1
        if current_sum > 1:
            return None
        
        # Determine range for next denominator
        # 1/d <= 1 - current_sum
        # d >= 1 / (1 - current_sum)
        remaining = 1 - current_sum
        if remaining <= 0:
            return None
        
        min_d = max(last_denom + 1, int(1 / remaining) + 1)
        
        # Upper bound for d:
        # We need sum of remaining (k-idx-1) terms to be at least 1 - (current_sum + 1/d)
        # Max sum of remaining (k-idx-1) terms with denominators > d
        # If 1/d + max_sum_remaining < 1, then d is too large?
        # Actually, if 1/d is too small, we can't reach 1.
        # 1/d >= 1 - (current_sum + max_sum_remaining)
        
        # Iterate d
        # Limit d to a reasonable range to avoid infinite loops
        # d cannot be too large, otherwise 1/d is too small.
        # 1/d <= remaining
        # d >= 1/remaining
        
        # Also, d must be such that we can fill the rest.
        # 1/d + sum_{j=1}^{k-idx-1} 1/(d+j) >= remaining
        # This is a tight constraint.
        
        # Simple upper bound: d < last_denom + 1000000 (arbitrary large)
        # But we can be smarter.
        # If k-idx == 1, then 1/d = remaining => d = 1/remaining
        if k - idx == 1:
            d_val = int(1 / remaining)
            if Fraction(1, d_val) == remaining and d_val > last_denom:
                d.append(d_val)
                if dfs(idx + 1, current_sum + Fraction(1, d_val), d_val):
                    return d[:]
                d.pop()
            return None
        
        # For k - idx > 1
        # d must be at least min_d
        # d must be such that 1/d + max_sum(k-idx-1, d+1) >= remaining
        # max_sum(m, start) = sum_{i=0}^{m-1} 1/(start+i)
        
        # Upper bound for d:
        # 1/d must be at least remaining - max_sum(k-idx-1, d+1)
        # This is hard to solve analytically.
        # Just iterate d from min_d upwards.
        
        # How far to iterate?
        # If d is large, 1/d is small.
        # The maximum sum of the remaining k-idx-1 terms is approx (k-idx-1)/d.
        # So 1/d + (k-idx-1)/d >= remaining
        # k/d >= remaining
        # d <= k / remaining
        
        max_d = int(k / remaining) + 10
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            new_sum = current_sum + Fraction(1, d_val)
            if new_sum > 1:
                continue
                
            d.append(d_val)
            res = dfs(idx + 1, new_sum, d_val)
            if res:
                return res
            d.pop()
            
        return None

    return dfs(0, Fraction(0), 4)

def _fallback():
    # Replicate seed logic
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
