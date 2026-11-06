"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Try to find solutions with increasing number of terms
    # For each k, do a bounded DFS to find k distinct denominators >= 5 summing to 1
    
    def find_solution(k, max_denom=10000000, min_denom=5):
        """Find k distinct denominators in [min_denom, max_denom] whose reciprocals sum to 1."""
        # Use DFS with pruning
        best = [None]
        
        def dfs(remaining, min_d, chosen, depth):
            if time.time() > deadline:
                return
            if depth == k:
                if remaining == 0:
                    best[0] = chosen[:]
                    return
                return
            if remaining < 0:
                return
            
            # Pruning: even if we take the smallest possible denominators for remaining terms,
            # can we reach the target?
            # Max possible sum with (k-depth) terms starting from min_d:
            # 1/min_d + 1/(min_d+1) + ... but that's upper bound
            # Min possible sum: 1/max_denom * (k-depth)
            
            # Lower bound on what we need: remaining
            # Upper bound on what we can add: sum of 1/d for d from min_d to min_d+(k-depth-1)
            # But that's loose. Better: the max we can add is 1/min_d + 1/(min_d+1) + ...
            
            # Simple pruning: if remaining * (k - depth) * min_d < 1 (rough), might be too small
            # Actually: max sum of (k-depth) unit fractions with denominators >= min_d
            # is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+k-depth-1)
            # If that's less than remaining, prune.
            
            max_possible = 0.0
            for i in range(k - depth):
                max_possible += 1.0 / (min_d + i)
            if max_possible < remaining - 1e-15:
                return
            
            # Min sum: (k-depth) / max_denom
            if (k - depth) / max_denom > remaining + 1e-15:
                return
            
            # The next denominator d must satisfy:
            # 1/d >= remaining / (k - depth)  =>  d <= (k-depth) / remaining
            # Also 1/d <= remaining  =>  d >= 1/remaining
            # And d >= min_d
            
            if remaining <= 0:
                return
                
            max_d = int((k - depth) / remaining)
            if max_d > max_denom:
                max_d = max_denom
            
            # d must be at least min_d
            # Also, we need enough room: 1/d + (k-depth-1)/d > remaining is not quite right
            # Better: 1/d <= remaining (otherwise overshoot)
            # And 1/d >= remaining / (k-depth) (otherwise even max won't reach)
            
            d_min = max(min_d, int(1.0 / remaining) + 1)
            d_max = min(max_d, max_denom)
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return
                new_remaining = remaining - 1.0 / d
                if new_remaining < -1e-12:
                    continue
                if new_remaining < 1e-12:
                    if k - depth == 1:
                        best[0] = chosen + [d]
                        return
                    continue
                chosen.append(d)
                dfs(new_remaining, d + 1, chosen, depth + 1)
                chosen.pop()
                if best[0] is not None:
                    return
        
        dfs(1.0, min_denom, [], 0)
        return best[0]
    
    # Try k=2, 3, 4, ... until we find a solution
    for k in range(2, 15):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            # Verify
            s = sum(1.0 / d for d in result)
            if abs(s - 1.0) < 1e-9 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use a known representation or the DP approach
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 + ... 
    # Let's use a simpler known one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520
    # Check: 1/5+1/6+1/7 = 42/210+35/210+30/210 = 107/210
    # 1 - 107/210 = 103/210
    # 1/42 = 5/210, so 103/210 - 5/210 = 98/210 = 7/15
    # 1/140... this is getting complicated.
    
    # Use the DP approach as fallback
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
    
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
