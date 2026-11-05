import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    best = [target]
    best_max = target
    
    def get_divisors(n):
        """Generate all divisors of n."""
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    def search(remaining, min_factor, current_factors, max_so_far, used_set):
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Check if we can terminate successfully
        if remaining == 1:
            if len(current_factors) >= 1:
                mx = max(current_factors) if current_factors else 0
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        # If we already have 12 factors, we can't add more
        if len(current_factors) >= 12:
            return
        
        # Pruning: if max_so_far >= best_max, no improvement
        if max_so_far >= best_max:
            return
        
        # Pruning: the remaining must be splittable into factors all >= min_factor
        # and > 12. The minimum possible factor is max(min_factor, 13)
        min_possible = max(min_factor, 13)
        
        # If remaining < min_possible and remaining != 1, we can't form a valid factor
        if remaining < min_possible:
            return
        
        # If remaining itself is a valid factor (>= min_possible, not used), we can take it
        # But we should also try splitting it further if we have room
        
        # Get divisors of remaining that are >= min_possible
        # To limit search, only consider divisors up to best_max
        divs = []
        for i in range(1, isqrt(remaining) + 1):
            if remaining % i == 0:
                d1 = i
                d2 = remaining // i
                if d1 >= min_possible and d1 not in used_set and d1 < best_max:
                    divs.append(d1)
                if d2 != d1 and d2 >= min_possible and d2 not in used_set and d2 < best_max:
                    divs.append(d2)
        
        # Also consider remaining itself
        if remaining >= min_possible and remaining not in used_set and remaining < best_max:
            divs.append(remaining)
        
        if not divs:
            return
        
        # Sort divisors: try smaller ones first (to keep factors balanced and allow more factors)
        # But also try ones close to geometric mean
        max_factors_left = 12 - len(current_factors)
        if max_factors_left > 1:
            geo_mean = remaining ** (1.0 / max_factors_left)
            # Sort by closeness to geo_mean, but prefer smaller to allow more splits
            divs.sort(key=lambda d: (abs(d - geo_mean), d))
        else:
            divs.sort()
        
        # Limit divisors to try
        max_divs = 150
        divs = divs[:max_divs]
        
        for d in divs:
            if d < min_possible:
                continue
            if d in used_set:
                continue
            
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            
            new_used = used_set | {d}
            new_factors = current_factors + [d]
            new_min = d + 1  # next factor must be > d for distinctness and ordering
            
            search(remaining // d, new_min, new_factors, new_max, new_used)
            
            if time.time() - start_time > DEADLINE:
                return
    
    search(target, 13, [], 0, set())
    
    return best
