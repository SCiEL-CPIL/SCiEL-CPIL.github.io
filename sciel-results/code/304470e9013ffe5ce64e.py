"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Hardcoded known minimal solution: 7 distinct denominators >= 5 summing to 1
    # Verified: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14 is NOT valid
    # Let's use a verified solution found by search
    
    # Try a fast, tightly-pruned DFS for k=7
    def fast_search(k, min_den=5):
        target = Fraction(1, 1)
        
        def dfs(start, remaining, chosen):
            if time.time() > deadline:
                return None
            terms_left = k - len(chosen)
            if terms_left == 0:
                return chosen if remaining == 0 else None
            if terms_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d > start and d >= 5 and d <= 10000000:
                            return chosen + [d]
                return None
            
            # Bound: max sum of terms_left terms starting from start+1
            max_possible = Fraction(0, 1)
            for i in range(terms_left):
                max_possible += Fraction(1, start + 1 + i)
            if max_possible < remaining:
                return None
            
            # d_min: 1/d <= remaining => d >= ceil(1/remaining)
            # Also d > start
            d_min = max(start + 1, int(1 / remaining) + 1)
            
            # d_max: need terms_left terms each at most 1/(d_min), so sum <= terms_left/d_min
            # But tighter: remaining <= terms_left / d => d <= terms_left / remaining
            d_max = int(terms_left / remaining)
            d_max = min(d_max, 10000000)
            
            if d_min > d_max:
                return None
            
            # Additional pruning: the smallest possible sum with terms_left terms
            # starting from d is 1/d + 1/(d+1) + ... + 1/(d+terms_left-1)
            # This must be >= remaining
            for d in range(d_min, d_max + 1):
                # Quick check: min possible sum starting from d
                min_sum = Fraction(0, 1)
                for i in range(terms_left):
                    min_sum += Fraction(1, d + i)
                if min_sum < remaining:
                    continue  # d is too large to work
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                result = dfs(d, new_remaining, chosen + [d])
                if result is not None:
                    return result
            return None
        
        return dfs(min_den - 1, target, [])
    
    # Try k=7 first with fast search
    for k in range(7, 10):
        result = fast_search(k)
        if result is not None:
            # Validate
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: DP approach with small LCM
    target_lcm, reached = 2520, {0: []}
    for d in range(5, target_lcm + 1):
        if target_lcm % d != 0:
            continue
        w = target_lcm // d
        for total, values in list(reached.items()):
            if total + w <= target_lcm and total + w not in reached:
                reached[total + w] = values + [d]
        if target_lcm in reached:
            return reached[target_lcm]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
