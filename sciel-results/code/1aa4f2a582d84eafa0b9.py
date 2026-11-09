"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    # Theoretical minimum for distinct denominators >= 5 is 8.
    # We use a DFS with greedy ordering (smallest denominators first) to find a solution of length 8.
    # This is effectively a greedy search that backtracks only if necessary.
    
    target = Fraction(1, 1)
    min_den = 5
    max_den = 10000000
    max_terms = 8  # We know 8 is the minimum, so we search for exactly 8 terms
    
    def dfs(remaining, min_d, current_list, depth):
        if depth == max_terms:
            if remaining == 0:
                return current_list
            return None
        
        # Pruning: If the maximum possible sum of the remaining terms (using smallest available denominators) is less than 'remaining', we can't reach it.
        # Max sum of (max_terms - depth) terms starting from min_d:
        # This is an upper bound. If sum_{i=0}^{k-1} 1/(min_d+i) < remaining, then impossible.
        # However, calculating this exactly every step might be slow. 
        # A simpler bound: 1/min_d is the largest term. 
        # If remaining > (max_terms - depth) * (1/min_d), it's impossible? No, that's an overestimate of the sum.
        # Actually, if remaining is very large, we need small denominators.
        
        # Greedy lower bound for next denominator:
        # 1/d <= remaining => d >= 1/remaining
        # Also d >= min_d
        # Also we need to be able to fill the rest.
        
        lower_bound = max(min_d, 1 / remaining if remaining > 0 else min_d)
        # Ensure integer
        if lower_bound < min_d:
            lower_bound = min_d
        
        # Upper bound for d:
        # We need to leave enough sum for the remaining terms.
        # The smallest sum for (max_terms - depth - 1) terms starting from d+1 is roughly (max_terms - depth - 1) / (d + max_terms).
        # This is complex. Let's just iterate.
        
        # To keep it fast, we limit the search range.
        # Since we are looking for 8 terms, and min_d starts at 5, the denominators won't get huge until the last few terms.
        
        for d in range(int(lower_bound), max_den + 1):
            if 1 / d > remaining:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break # Since d is increasing, 1/d is decreasing, so new_remaining will only get larger (less negative? No, 1/d decreases, so subtracting less means new_remaining increases. Wait.
                # remaining is fixed. d increases -> 1/d decreases -> remaining - 1/d increases.
                # So if new_remaining < 0, it means 1/d > remaining.
                # As d increases, 1/d gets smaller, so eventually 1/d <= remaining.
                # So we should NOT break, we should continue until 1/d <= remaining.
                # But we started from lower_bound where 1/d <= remaining.
                # So new_remaining should be >= 0.
                pass
            
            # Pruning: Can we fill the rest?
            # Max sum of remaining terms (max_terms - depth - 1) terms starting from d+1.
            # If this max sum < new_remaining, then this d is too small (we need smaller denominators, but we are increasing d).
            # Wait, if d is small, 1/d is large.
            # If we pick a small d, we subtract a lot.
            # If we pick a large d, we subtract a little.
            # If new_remaining is very large, we need to pick a small d.
            # If new_remaining is very small, we need to pick a large d.
            
            # Heuristic: If new_remaining == 0 and depth+1 < max_terms, we have too few terms.
            # We must use exactly 8 terms.
            
            res = dfs(new_remaining, d + 1, current_list + [d], depth + 1)
            if res:
                return res
        
        return None

    result = dfs(target, min_den, [], 0)
    if result:
        return result
    
    # Fallback: Return the known 8-term solution if DFS fails (shouldn't happen)
    # One known solution: [5, 6, 7, 8, 9, 10, 12, 14] sums to 1 + 1/2520.
    # We need a valid one.
    # The DP in the seed likely found one.
    # Let's just return the seed's logic if this fails, but the prompt says "do not reuse".
    # I will provide a hardcoded valid 8-term solution found via quick manual search or trust the DFS.
    
    # Actually, the set [5, 6, 7, 8, 9, 10, 12, 14] is close.
    # Let's try [5, 6, 7, 8, 9, 10, 13, 14]?
    # 1/12 -> 1/13. Sum decreases by 1/12 - 1/13 = 1/156.
    # Previous excess was 1/2520.
    # 1/156 > 1/2520. So sum becomes < 1.
    
    # The DFS should find it.
    raise RuntimeError("No 8-term solution found")
# EVOLVE-BLOCK-END
