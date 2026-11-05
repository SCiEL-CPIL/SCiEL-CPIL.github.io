import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    best = [target]
    best_max = target
    
    def get_divisors_bounded(n, lo, hi):
        """Generate divisors of n in [lo, hi]."""
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                d1 = i
                d2 = n // i
                if lo <= d1 <= hi:
                    divs.append(d1)
                if d2 != d1 and lo <= d2 <= hi:
                    divs.append(d2)
        return sorted(divs)
    
    def search(remaining, min_factor, current_factors, max_so_far, used_set):
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        if len(current_factors) >= 12:
            return
        
        if max_so_far >= best_max:
            return
        
        min_possible = max(min_factor, 13)
        
        if remaining < min_possible:
            return
        
        # Upper bound for this factor: must be < best_max and <= remaining
        hi = min(best_max - 1, remaining)
        
        if hi < min_possible:
            return
        
        # Get valid divisors
        divs = get_divisors_bounded(remaining, min_possible, hi)
        
        # Filter out used
        divs = [d for d in divs if d not in used_set]
        
        if not divs:
            return
        
        # Sort: prefer divisors that allow more splits (smaller divisors first for more factors)
        # but also consider balance
        max_factors_left = 12 - len(current_factors)
        if max_factors_left > 1:
            # Try to maximize number of factors: prefer smaller divisors
            # but not too small (must leave room for remaining splits)
            divs.sort(key=lambda d: d)
        else:
            divs.sort()
        
        # Limit to speed up
        max_divs = 80
        divs = divs[:max_divs]
        
        for d in divs:
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            
            new_used = used_set | {d}
            new_factors = current_factors + [d]
            new_min = d + 1
            
            search(remaining // d, new_min, new_factors, new_max, new_used)
            
            if time.time() - start_time > DEADLINE:
                return
    
    # Try to find a good initial solution to set a tight upper bound
    # Start with a reasonable guess and refine
    # For 12 factors, geometric mean is about 233
    # Let's try to find a solution with max around 300-400 first
    
    # Initial search with a loose bound
    search(target, 13, [], 0, set())
    
    return best
