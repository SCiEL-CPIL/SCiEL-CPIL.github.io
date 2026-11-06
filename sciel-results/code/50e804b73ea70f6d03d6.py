"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    def dfs(r, last_d, max_d, path, depth_limit=10):
        if r == 0:
            return path
        if not r: # r < 0
            return None
        if len(path) >= depth_limit:
            return None
            
        # 1/d <= r  => d >= 1/r
        min_d = max(last_d + 1, (r.denominator + r.numerator - 1) // r.numerator)
        
        if min_d > max_d:
            return None
            
        # Pruning: Check if r can be formed by sum of reciprocals of distinct ints in [min_d, max_d]
        # Max sum is approx sum_{i=min_d}^{max_d} 1/i
        # If r is greater than this sum, impossible.
        # For small ranges, we can sum exactly. For large, approximate.
        # But r is small (<= 5/31). The sum of reciprocals from 7 to 100 is quite large (~2).
        # So this pruning is not very tight for small r.
        
        # Another pruning: 
        # If we use k terms, the max sum is sum of 1/min_d, 1/(min_d+1)...
        # The min sum is sum of 1/max_d, 1/(max_d-1)...
        
        for d in range(min_d, max_d + 1):
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            # Heuristic: Try larger denominators first? 
            # If we want to minimize max denominator, we are fixing max_d.
            # Inside, we just need ANY solution.
            # Trying smaller d first might lead to dead ends faster?
            # Or trying d close to 1/r?
            
            res = dfs(new_r, d, max_d, path + [d], depth_limit)
            if res:
                return res
        return None

    # Iterate max_denominator from lower bound upwards
    # Lower bound: 1/d_1 <= 5/31 => d_1 >= 7.
    # But we need at least 2 terms? No, 1 term impossible.
    # Start from 32 to be safe, or 7.
    # Let's start from 32 because 5/31 is close to 1/6.
    # If we start from 7, the search space is larger.
    # Let's start from 10.
    
    for max_d in range(10, 2000):
        res = dfs(target, 0, max_d, [])
        if res:
            return res
            
    # Fallback to greedy if no solution found (should not happen)
    r, values = target, []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
