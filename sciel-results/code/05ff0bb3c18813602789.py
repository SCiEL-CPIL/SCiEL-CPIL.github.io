from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = 55.0
    
    def dfs(start_d, remaining, target, current):
        if time.time() > deadline:
            return None
        
        if remaining == 0:
            if target == 0:
                return current
            return None
        
        # Pruning 1: Max sum
        # Sum of 'remaining' terms with smallest possible denominators (start_d, start_d+1, ...)
        # is at most remaining / start_d
        if target > Fraction(remaining, start_d):
            return None
        
        # Pruning 2: Min sum
        # Sum of 'remaining' terms with largest possible denominators (10^7, ...)
        # is at least remaining / 10000000
        # This is a very weak bound, but:
        # If target < remaining / 10000000, we can't reach it?
        # No, we can use even larger denominators? No, max is 10^7.
        # So min sum is sum of 1/(10^7 - i).
        # If target is smaller than this, impossible.
        # But target is usually not that small.
        
        # Determine range for d
        # 1/d <= target => d >= 1/target
        min_d = start_d
        if target > 0:
            inv_target = Fraction(1, 1) / target
            # ceil(inv_target)
            num, den = inv_target.numerator, inv_target.denominator
            ceil_val = (num + den - 1) // den
            if ceil_val > min_d:
                min_d = ceil_val
        
        # Upper bound
        # target - 1/d <= (remaining - 1) / (d + 1)
        R = target
        C = Fraction(remaining - 1, 1) - R
        # Rd^2 - C*d - 1 <= 0
        # Use float for bound
        R_f = float(R)
        C_f = float(C)
        if R_f == 0:
            return None
        disc = C_f * C_f + 4 * R_f
        if disc < 0:
            return None
        max_d_f = (C_f + disc ** 0.5) / (2 * R_f)
        max_d = int(max_d_f) + 2
        max_d = min(max_d, 10000000)
        
        if min_d > max_d:
            return None
        
        for d in range(min_d, max_d + 1):
            new_target = target - Fraction(1, d)
            if new_target < 0:
                continue # Should not happen if min_d is correct, but safe
            
            res = dfs(d + 1, remaining - 1, new_target, current + [d])
            if res is not None:
                return res
        
        return None

    for k in range(8, 15):
        result = dfs(5, k, Fraction(1, 1), [])
        if result is not None:
            return result
        if time.time() > deadline - 5:
            break

    raise RuntimeError("No solution found")
