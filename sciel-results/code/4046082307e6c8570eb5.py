"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    best_max = 10000000
    best_denoms = None
    
    # Search for solutions of form 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a must be >= 7
    # We iterate a up to a limit where 1/a + 1/(a+1) + 1/(a+2) < 5/31
    # or just a reasonable bound like 1000
    
    for a in range(7, 100):
        if time.time() - start_time > 50:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        
        # 1/b + 1/c = rem = N/D
        # (N*b - D)*(N*c - D) = D^2
        N = rem.numerator
        D = rem.denominator
        
        # b > D/N
        # c > b
        # (N*b - D) must be a divisor of D^2
        # Let X = N*b - D, Y = N*c - D
        # X * Y = D^2
        # X < Y
        # X >= N*(D//N + 1) - D (since b > D/N)
        # X < D (since Y > X implies X^2 < D^2)
        
        # Iterate divisors X of D^2
        # To be efficient, we can just check b in range
        # b_min = D // N + 1
        # b_max: since 1/b > rem/2 => b < 2*D/N
        b_min = D // N + 1
        b_max = (2 * D) // N
        
        if b_max < b_min:
            continue
            
        for b in range(b_min, b_max + 1):
            # Check if c is integer
            # 1/c = rem - 1/b = (N*b - D) / (D*b)
            num = N * b - D
            if num <= 0:
                continue
            den = D * b
            if den % num == 0:
                c = den // num
                if c > b:
                    max_d = c
                    if max_d < best_max:
                        best_max = max_d
                        best_denoms = [a, b, c]
    
    # If no 3-term solution found (unlikely), fallback to greedy
    if best_denoms is None:
        r = target
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals
        
    return best_denoms
# EVOLVE-BLOCK-END
