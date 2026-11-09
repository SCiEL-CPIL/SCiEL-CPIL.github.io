import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Precompute harmonic numbers up to 100
    MAX_B = 100
    H = [0.0] * (MAX_B + 1)
    for i in range(1, MAX_B + 1):
        H[i] = H[i-1] + 1.0/i
        
    def dfs(rem, min_d, B, current):
        if rem == 0:
            return current
            
        # Pruning: if rem is greater than the sum of all possible remaining denominators
        # Sum of 1/i for i in [min_d, B]
        if min_d <= B:
            max_possible = H[B] - H[min_d-1]
            if rem > max_possible:
                return None
        else:
            return None

        # Lower bound for next d
        # 1/d <= rem  => d >= 1/rem
        lb = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        if lb > B:
            return None
            
        for d in range(lb, B + 1):
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                continue
            
            res = dfs(next_rem, d + 1, B, current + [d])
            if res is not None:
                return res
        return None

    for B in range(1, MAX_B + 1):
        if time.time() - start_time > 50:
            break
        sol = dfs(target, 1, B, [])
        if sol is not None:
            return sol
            
    return [8, 31, 248]
