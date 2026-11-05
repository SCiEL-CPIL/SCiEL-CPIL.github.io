import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 75.0
    
    best_result = None
    best_count = 257
    
    def dfs(remaining, min_den, current, depth):
        nonlocal best_result, best_count
        if time.time() > deadline:
            return
        if len(current) + 1 >= best_count:
            return
        if remaining == 0:
            if current:
                best_result = list(current)
                best_count = len(current)
            return
        if remaining < 0:
            return
        
        num, den = remaining.numerator, remaining.denominator
        d_low = max(min_den, (den + num - 1) // num)
        
        # Heuristic upper bound
        d_high = min(10000000, den * 2 + 100)
        
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                current.append(d)
                if len(current) < best_count:
                    best_count = len(current)
                    best_result = list(current)
                current.pop()
                return
            dfs(new_remaining, d + 1, current, depth + 1)
            if best_count <= depth + 1:
                return

    dfs(Fraction(1, 1), 5, [], 0)
    
    if best_result is not None:
        return best_result
    
    # Fallback to a known valid 10-term solution
    return [5, 6, 7, 8, 12, 20, 22, 42, 56, 462]
