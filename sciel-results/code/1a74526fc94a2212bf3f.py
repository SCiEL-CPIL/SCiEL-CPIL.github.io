import time
from fractions import Fraction
import functools

def construct():
    start_time = time.time()
    limit_time = 55.0  # Safety margin before 60s limit
    
    # Memoization cache: (remaining, start_denom, terms_left) -> bool (feasible or not)
    # We use this to avoid re-exploring the same state
    cache = {}
    
    def is_feasible(remaining, start_denom, terms_left):
        """Check if it's possible to represent 'remaining' as sum of 'terms_left' distinct
        unit fractions with denominators >= start_denom."""
        if remaining == 0:
            return terms_left == 0
        if terms_left == 0:
            return remaining == 0
        if remaining < 0:
            return False
        
        key = (remaining, start_denom, terms_left)
        if key in cache:
            return cache[key]
        
        # Pruning: max possible sum
        if terms_left <= 20:
            max_s = Fraction(0, 1)
            for i in range(start_denom, start_denom + terms_left):
                max_s += Fraction(1, i)
            if max_s < remaining:
                cache[key] = False
                return False
        
        # Min denominator for next term
        min_d = max(start_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                result = d >= min_d and d <= 10000000
            else:
                result = False
            cache[key] = result
            return result
        
        # Upper bound for d
        max_d = min(10000000, (terms_left * remaining.denominator) // remaining.numerator + 1)
        
        for d in range(min_d, max_d + 1):
            if time.time() - start_time > limit_time:
                return False
            next_rem = remaining - Fraction(1, d)
            if next_rem < 0:
                continue
            if is_feasible(next_rem, d + 1, terms_left - 1):
                cache[key] = True
                return True
        
        cache[key] = False
        return False
    
    def find_solution(remaining, start_denom, terms_left, current):
        """Find an actual solution given that one exists."""
        if remaining == 0:
            return current[:]
        if terms_left == 0:
            return None
        
        min_d = max(start_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d and d <= 10000000:
                    return current + [d]
            return None
        
        max_d = min(10000000, (terms_left * remaining.denominator) // remaining.numerator + 1)
        
        for d in range(min_d, max_d + 1):
            if time.time() - start_time > limit_time:
                return None
            next_rem = remaining - Fraction(1, d)
            if next_rem < 0:
                continue
            if is_feasible(next_rem, d + 1, terms_left - 1):
                result = find_solution(next_rem, d + 1, terms_left - 1, current + [d])
                if result is not None:
                    return result
        
        return None
    
    # Try increasing k from 8 upward
    for k in range(8, 20):
        if time.time() - start_time > limit_time:
            break
        # Check feasibility first
        if is_feasible(Fraction(1, 1), 5, k):
            # Find the actual solution
            res = find_solution(Fraction(1, 1), 5, k, [])
            if res is not None:
                # Verify
                if sum(Fraction(1, x) for x in res) == 1 and all(5 <= x <= 10000000 for x in res) and len(set(res)) == len(res):
                    return res
    
    # Fallback
    raise RuntimeError("No solution found")
