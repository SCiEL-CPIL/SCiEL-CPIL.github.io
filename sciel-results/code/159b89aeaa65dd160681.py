import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    max_den = 10_000_000
    
    # Pre-compute harmonic number approximations for quick max_sum
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        """Approximate harmonic number H(n) = sum(1/i for i=1..n)"""
        if n <= 0:
            return 0.0
        if n < 1000:
            return sum(1.0/i for i in range(1, n+1))
        return math.log(n) + gamma + 1.0/(2*n)
    
    def sum_recip_approx(a, b):
        """Approximate sum(1/i for i=a..b)"""
        if a > b:
            return 0.0
        if b - a < 1000:
            return sum(1.0/i for i in range(a, b+1))
        return harmonic_approx(b) - harmonic_approx(a-1)
    
    def find_solution(k):
        """
        Find k distinct denominators d1 < ... < dk in [5, 10^7] 
        such that sum(1/di) = 1.
        Returns list of denominators or None.
        """
        target = Fraction(1, 1)
        
        def dfs(rem, min_den, current, depth):
            if time.time() > deadline:
                return None
            
            # Base case
            if depth == k:
                if rem == 0:
                    return current[:]
                return None
            
            # Pruning:
            # 1. rem must be positive
            if rem <= 0:
                return None
            
            remaining = k - depth
            
            # 2. Max possible sum of remaining terms
            max_sum = sum_recip_approx(min_den, min_den + remaining - 1)
            if float(rem) > max_sum:
                return None
            
            # 3. Min possible sum of remaining terms
            min_sum = remaining / max_den
            if float(rem) < min_sum:
                return None
            
            # 4. Bounds for next denominator d
            d_min = max(min_den, math.ceil(1.0 / float(rem)))
            
            d = d_min
            while d <= max_den:
                if time.time() > deadline:
                    return None
                
                frac_d = Fraction(1, d)
                if frac_d > rem:
                    break
                
                new_rem = rem - frac_d
                
                if new_rem <= 0:
                    if new_rem == 0 and remaining == 1:
                        return current + [d]
                    d += 1
                    continue
                
                # Check if we can form new_rem with remaining-1 terms > d
                if remaining > 1:
                    max_rest = sum_recip_approx(d + 1, d + remaining)
                    if float(new_rem) > max_rest:
                        return None
                    
                    min_rest = (remaining - 1) / max_den
                    if float(new_rem) < min_rest:
                        d += 1
                        continue
                
                result = dfs(new_rem, d + 1, current + [d], depth + 1)
                if result:
                    return result
                
                d += 1
                
            return None

        return dfs(target, 5, [], 0)

    # Minimum k is 8 (sum of 1/5..1/12 > 1, sum of 1/5..1/11 < 1)
    for k in range(8, 16):
        sol = find_solution(k)
        if sol:
            s = Fraction(0, 1)
            for d in sol:
                s += Fraction(1, d)
            if s == 1 and len(set(sol)) == k and all(5 <= d <= max_den for d in sol):
                return sol
        
        if time.time() > deadline - 5.0:
            break

    raise RuntimeError("No solution found within time limit")
