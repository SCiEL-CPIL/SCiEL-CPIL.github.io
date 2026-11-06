import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute harmonic numbers H_n = sum_{i=1}^n 1/i as Fraction or float?
    # Float might be risky for exact comparison, but for pruning "if max_possible < rem" it's okay if we have margin.
    # Or use Fraction for safety.
    H = [Fraction(0, 1)] * 248
    H[0] = Fraction(0, 1)
    for i in range(1, 248):
        H[i] = H[i-1] + Fraction(1, i)
        
    def search_max_denom(C):
        """
        Returns a list of denominators summing to 5/31 with all d <= C, or None.
        """
        def backtrack(num, den, min_d, chosen):
            if num == 0:
                if chosen:
                    return chosen
                return None
            
            if num < 0:
                return None
                
            # If num/den is a unit fraction 1/d
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= C:
                    return chosen + [d]
                return None
            
            min_d_eff = max(min_d, den // num + 1)
            
            if min_d_eff > C:
                return None
                
            # Pruning:
            # Max sum available from d+1 to C is H[C] - H[d]
            # We need num/den - 1/d <= H[C] - H[d]
            # => num/den <= 1/d + H[C] - H[d]
            
            for d in range(min_d_eff, C + 1):
                # Check if 1/d < num/den
                if num * d <= den:
                    continue
                
                # Pruning check
                # rem_after = num/den - 1/d
                # max_avail = H[C] - H[d]
                # if rem_after > max_avail: continue (or break if d increases? 
                # As d increases, 1/d decreases, so rem_after increases.
                # H[C] - H[d] decreases.
                # So if it fails for some d, it will fail for all larger d?
                # Let's check.
                # f(d) = num/den - 1/d - (H[C] - H[d])
                # We want f(d) <= 0.
                # As d increases, -1/d becomes less negative (increases).
                # -H[d] becomes more negative (decreases).
                # H[d] increases, so -H[d] decreases.
                # The derivative of -1/d is 1/d^2 > 0.
                # The derivative of -H[d] is approx -1/d < 0.
                # Net change: 1/d^2 - 1/d < 0 for d > 1.
                # So the condition becomes "tighter" (more likely to fail) as d increases?
                # Let's test.
                # d=10, C=20. H[20]-H[10] ~ 0.3.
                # d=11, C=20. H[20]-H[11] ~ 0.29.
                # rem_after increases slightly.
                # So if it fails for d, it likely fails for d+1.
                # We can break.
                
                rem_num = num * d - den
                rem_den = den * d
                # rem = rem_num / rem_den
                # max_avail = H[C] - H[d]
                # if rem_num/rem_den > H[C] - H[d]:
                #   break
                
                # Compare rem_num/rem_den > H[C] - H[d]
                # rem_num * (H[C] - H[d]).denominator > rem_den * (H[C] - H[d]).numerator
                # This is getting complicated with Fractions.
                # Let's just use float approximation for pruning, and verify later.
                # Or just skip this pruning if it's too complex.
                # The simple bound d > den/num is strong enough for small C.
                
                new_num = num * d - den
                new_den = den * d
                
                g = math.gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                res = backtrack(new_num, new_den, d + 1, chosen + [d])
                if res is not None:
                    return res
                    
            return None
        
        return backtrack(5, 31, 1, [])

    for C in range(1, 248):
        if time.time() > deadline - 5:
            break
        sol = search_max_denom(C)
        if sol is not None:
            s = sum(Fraction(1, d) for d in sol)
            if s == Fraction(5, 31):
                return sorted(sol)
                
    return [8, 31, 248]
