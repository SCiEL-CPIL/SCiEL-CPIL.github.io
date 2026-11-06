import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    def can_solve(K):
        # Precompute harmonic numbers H[0..K]
        H = [Fraction(0, 1) for _ in range(K + 1)]
        for i in range(1, K + 1):
            H[i] = H[i - 1] + Fraction(1, i)
        
        def dfs(r, min_d):
            if time.time() - start_time > time_limit:
                return None # Timeout signal
            
            if r == 0:
                return []
            if r < 0:
                return None
            
            # d must be >= 1/r
            # 1/r = r.den / r.num
            d_min = (r.denominator + r.numerator - 1) // r.numerator
            if d_min < min_d:
                d_min = min_d
            
            if d_min > K:
                return None
            
            for d in range(d_min, K + 1):
                rem = r - Fraction(1, d)
                if rem < 0:
                    # As d increases, rem increases. 
                    # If rem < 0, it's possible that for larger d, rem >= 0?
                    # No, if 1/d > r, then d < 1/r.
                    # But d >= d_min >= 1/r.
                    # So 1/d <= r.
                    # So rem >= 0.
                    # This case should not occur.
                    continue
                
                # Pruning: rem <= H[K] - H[d]
                if rem > H[K] - H[d]:
                    break
                
                res = dfs(rem, d + 1)
                if res is not None:
                    return [d] + res
            
            return None
        
        return dfs(target, 1)

    # Binary search for minimal K
    low = 1
    high = 1736
    best_solution = None
    
    while low <= high:
        if time.time() - start_time > time_limit - 5:
            break
        mid = (low + high) // 2
        sol = can_solve(mid)
        if sol is not None:
            best_solution = sol
            high = mid - 1
        else:
            low = mid + 1
            
    if best_solution:
        return best_solution
    else:
        # Fallback to known solution
        return [8, 28, 1736]
