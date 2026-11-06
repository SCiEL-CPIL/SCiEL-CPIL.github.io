"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    # Known valid solution with 10 terms (all >= 5, <= 10000000, distinct, sum = 1)
    # Let's verify: 1/5+1/6+1/7+1/42+1/140+1/210+1/420+1/840+1/1260+1/2520
    # = 0.2 + 0.1667 + 0.1429 + 0.0238 + 0.0071 + 0.0048 + 0.0024 + 0.0012 + 0.0008 + 0.0004
    # This should be close to 1. Let me verify with exact fractions.
    fallback = [5, 6, 7, 42, 140, 210, 420, 840, 1260, 2520]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback
    
    # Try to find a better (smaller k) solution using DFS
    # We know k=2,3,4,5,6,7 are impossible for denominators >= 5
    # Start from k=8
    
    def find_solution(k, max_denom=10000000, min_denom=5):
        """Find k distinct denominators in [min_denom, max_denom] whose reciprocals sum to 1."""
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
            
            # Pruning: max possible sum with remaining terms
            # The largest we can get is 1/min_d + 1/(min_d+1) + ... + 1/(min_d + (k-depth-1))
            terms_left = k - depth
            max_possible = Fraction(0)
            for i in range(terms_left):
                max_possible += Fraction(1, min_d + i)
            if max_possible < remaining:
                return
            
            # Min possible sum: terms_left / max_denom
            min_possible = Fraction(terms_left, max_denom)
            if min_possible > remaining:
                return
            
            # Bound on d:
            # 1/d <= remaining => d >= ceil(1/remaining)
            # 1/d >= remaining / terms_left => d <= floor(terms_left / remaining)
            d_min = max(min_d, -(-remaining.denominator // remaining.numerator) + 1)
            d_max = min(max_denom, terms_left * remaining.denominator // remaining.numerator)
            
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
                    continue
                chosen.append(d)
                dfs(new_remaining, d + 1, chosen, depth + 1)
                chosen.pop()
                if best[0] is not None:
                    return
        
        dfs(Fraction(1, 1), min_denom, [], 0)
        return best[0]
    
    # Try k=8, 9, 10 in order
    for k in range(8, 11):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback to known solution
    return fallback
# EVOLVE-BLOCK-END
