"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    # Try to find solutions with increasing number of terms
    # For each k, do a bounded DFS to find k distinct denominators >= 5 summing to 1
    
    def find_solution(k, max_denom=10000000, min_denom=5):
        """Find k distinct denominators in [min_denom, max_denom] whose reciprocals sum to 1."""
        # Use DFS with pruning using exact Fraction arithmetic
        best = [None]
        
        def dfs(remaining, min_d, chosen, depth):
            if time.time() > deadline:
                return
            if depth == k:
                if remaining == 0:
                    best[0] = chosen[:]
                    return
                return
            if remaining <= 0:
                return
            
            terms_left = k - depth
            
            # Pruning: max possible sum with terms_left terms starting from min_d
            # is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms_left-1)
            # If that's less than remaining, prune.
            max_possible = Fraction(0)
            for i in range(terms_left):
                max_possible += Fraction(1, min_d + i)
            if max_possible < remaining:
                return
            
            # Min sum: terms_left / max_denom (using smallest possible denominators)
            # Actually min sum is when denominators are as large as possible: terms_left / max_denom
            if Fraction(terms_left, max_denom) > remaining:
                return
            
            # The next denominator d must satisfy:
            # 1/d >= remaining / terms_left  =>  d <= terms_left / remaining
            # Also 1/d <= remaining  =>  d >= 1/remaining
            # And d >= min_d
            
            # d <= terms_left / remaining => d <= floor(terms_left / remaining)
            # Since remaining = p/q, terms_left/remaining = terms_left*q/p
            max_d = (terms_left * remaining.denominator) // remaining.numerator
            if max_d > max_denom:
                max_d = max_denom
            
            # d >= 1/remaining => d >= ceil(1/remaining) = ceil(remaining.denominator/remaining.numerator)
            # But also d >= min_d
            # 1/d <= remaining => d >= 1/remaining => d >= ceil(remaining.denominator / remaining.numerator)
            d_min_from_frac = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(min_d, d_min_from_frac)
            d_max = min(max_d, max_denom)
            
            if d_min > d_max:
                return
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    continue
                if new_remaining == 0:
                    if terms_left == 1:
                        best[0] = chosen + [d]
                        return
                    else:
                        continue
                chosen.append(d)
                dfs(new_remaining, d + 1, chosen, depth + 1)
                chosen.pop()
                if best[0] is not None:
                    return
        
        dfs(Fraction(1, 1), min_denom, [], 0)
        return best[0]
    
    # Try k=2, 3, 4, ... until we find a solution
    for k in range(2, 12):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            # Verify with exact arithmetic
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: known representation
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
    # Let me verify: LCM of 5,6,7,10,14,35,42,70,105,210 = 210
    # 1/5=42/210, 1/6=35/210, 1/7=30/210, 1/10=21/210, 1/14=15/210, 1/35=6/210, 1/42=5/210, 1/70=3/210, 1/105=2/210, 1/210=1/210
    # Sum = 42+35+30+21+15+6+5+3+2+1 = 160/210 = 16/21 ≠ 1
    
    # Use DP fallback with larger target
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
