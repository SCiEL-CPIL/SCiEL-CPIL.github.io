import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 5.0

    # Try to find 8 terms
    # Target: 1
    # Denoms >= 5, distinct
    # Sort denominators to avoid permutations
    
    found_8 = None
    found_9 = None
    
    # DFS for k terms
    def dfs(remaining_sum, min_denom, k, current):
        if k == 0:
            if remaining_sum == 0:
                return current[:]
            return None
        if remaining_sum <= 0:
            return None
            
        # Pruning: max possible sum with k terms starting from min_denom
        # 1/min_denom + 1/(min_denom+1) + ...
        # If this sum < remaining_sum, prune
        max_possible = 0
        for i in range(k):
            max_possible += Fraction(1, min_denom + i)
        if max_possible < remaining_sum:
            return None
            
        # Pruning: min possible sum with k terms (infinite denominators -> 0)
        # But we need distinct integers.
        # If remaining_sum is very small, we need large denominators.
        
        # Lower bound for next denominator
        # 1/d <= remaining_sum / k  => d >= k / remaining_sum
        # Also d > min_denom - 1
        # And d >= 5
        lower_bound = max(min_denom, int(k / remaining_sum) + 1)
        
        # Upper bound for next denominator
        # 1/d >= remaining_sum / k is not true, 1/d can be small.
        # But 1/d must be at least remaining_sum - (sum of next k-1 max terms)
        # This is complex.
        # Simple bound: d < 1/remaining_sum if k=1.
        # If k > 1, d can be large.
        
        # Iterate d from lower_bound
        # Limit search depth or d range to prevent infinite loop
        # If remaining_sum is small, d will be large.
        # 1/d < remaining_sum => d > 1/remaining_sum
        # If k=1, d = 1/remaining_sum must be integer.
        
        if k == 1:
            if remaining_sum.numerator == 1:
                d = remaining_sum.denominator
                if d >= min_denom and d >= 5:
                    return current + [d]
            return None
        
        # Heuristic limit for d to keep it fast
        # If d is too large, sum will be too small?
        # We already checked max_possible.
        # Let's limit d to min_denom + 1000 or similar for fast search?
        # No, d can be large.
        # But for k=8, denominators are usually small (< 50).
        
        for d in range(lower_bound, min_denom + 200): # Heuristic: check small d first
            if d < 5:
                continue
            # Check if 1/d <= remaining_sum
            if Fraction(1, d) > remaining_sum:
                continue
                
            # Recursive step
            new_remaining = remaining_sum - Fraction(1, d)
            res = dfs(new_remaining, d + 1, k - 1, current + [d])
            if res:
                return res
            if time.time() > deadline:
                return "TIMEOUT"
                
        return None

    # Search for 8
    res_8 = dfs(Fraction(1, 1), 5, 8, [])
    if res_8 and res_8 != "TIMEOUT":
        # Verify
        if sum(Fraction(1, d) for d in res_8) == 1:
            return res_8
            
    # Search for 9
    res_9 = dfs(Fraction(1, 1), 5, 9, [])
    if res_9 and res_9 != "TIMEOUT":
        if sum(Fraction(1, d) for d in res_9) == 1:
            return res_9
            
    # Fallback: Hardcoded 9-term solution if search fails
    # We need a valid one.
    # Let's use the one from the failed program's logic but corrected?
    # Or just return a known valid one.
    # Known valid 9-term: [5, 6, 7, 8, 9, 10, 11, 12, 13] is invalid (sum > 1).
    # Let's construct one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ...
    # Actually, let's just return the result of the search. If search fails, we are in trouble.
    # But the search should find one.
    
    # If all else fails, return a dummy valid set?
    # We can generate one using a simple algorithm if search fails.
    # But search should work.
    
    return res_9
