import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Try known solutions first for speed
    # Verify candidate solutions
    def validate_solution(denoms):
        if not (1 <= len(denoms) <= 256):
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(1, 1)
    
    # Try to find minimal solution
    # 4 terms impossible: 1/5+1/6+1/7+1/8 < 1
    # Try 5 terms first
    
    def search_minimal(k, min_den=5):
        """Search for k distinct denominators >= min_den summing to 1"""
        if time.time() > deadline:
            return None
        
        def dfs(start, remaining, chosen):
            if time.time() > deadline:
                return None
            
            terms_needed = k - len(chosen)
            
            if terms_needed == 0:
                if remaining == 0:
                    return chosen[:]
                return None
            
            if terms_needed == 1:
                # Need 1/d = remaining, so d = 1/remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d > start and d >= 5 and d <= 10000000:
                            return chosen + [d]
                return None
            
            # Pruning: max possible sum with terms_needed terms starting from start+1
            max_possible = Fraction(0, 1)
            for i in range(terms_needed):
                max_possible += Fraction(1, start + 1 + i)
            if max_possible < remaining:
                return None
            
            # d must satisfy: 1/d <= remaining => d >= ceil(1/remaining)
            # And: terms_needed/d >= remaining => d <= floor(terms_needed/remaining)
            if remaining <= 0:
                return None
            
            inv_rem = 1 / remaining
            d_min = max(start + 1, int(inv_rem) + (1 if inv_rem != int(inv_rem) else 0))
            d_max = int(terms_needed / remaining)
            d_max = min(d_max, 10000000)
            
            if d_min > d_max:
                return None
            
            # Additional pruning: even with smallest possible remaining terms,
            # can we reach the target?
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return None
                
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                
                # Check if remaining terms can sum to new_remaining
                if terms_needed - 1 > 0:
                    # Max sum of (terms_needed-1) terms starting from d+1
                    max_next = Fraction(0, 1)
                    for i in range(terms_needed - 1):
                        max_next += Fraction(1, d + 1 + i)
                    if max_next < new_remaining:
                        # d is too small, need larger d
                        continue
                    # Min sum check: (terms_needed-1)/d_next_max where d_next_max is large
                    # This is always positive, so no lower bound issue
                
                result = dfs(d, new_remaining, chosen + [d])
                if result is not None:
                    return result
            
            return None
        
        return dfs(min_den - 1, Fraction(1, 1), [])
    
    # Try increasing k from 5 (4 is impossible)
    for k in range(5, 12):
        if time.time() > deadline:
            break
        result = search_minimal(k)
        if result is not None:
            if validate_solution(result):
                return result
    
    # Fallback: use DP approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            sol = reached[target]
            if validate_solution(sol):
                return sol
    
    raise RuntimeError("no solution found")
