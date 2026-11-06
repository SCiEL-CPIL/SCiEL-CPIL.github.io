import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 45.0  # Safety margin for 60s limit

    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a must satisfy 1/a < 5/31 => a > 6.2 => a >= 7
    # 3/a > 5/31 => a < 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
            
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        
        # Solve 1/b + 1/c = r
        # r = N/D
        N = r.numerator
        D = r.denominator
        
        # (Nb-D)(Nc-D) = D^2
        # X = Nb-D, Y = Nc-D
        # X*Y = D^2
        # b = (X+D)/N, c = (Y+D)/N
        
        D2 = D * D
        
        # Find divisors of D2
        divisors = []
        i = 1
        while i * i <= D2:
            if D2 % i == 0:
                divisors.append(i)
                if i != D2 // i:
                    divisors.append(D2 // i)
            i += 1
        
        for X in divisors:
            if time.time() > deadline:
                break
                
            Y = D2 // X
            
            # Check congruence: X+D must be divisible by N
            if (X + D) % N != 0:
                continue
            if (Y + D) % N != 0:
                continue
            
            b = (X + D) // N
            c = (Y + D) // N
            
            # Ensure distinct and sorted (a < b < c) to avoid duplicates
            # and valid denominators
            if b <= a or c <= b:
                continue
            
            # Verify solution
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                current_max = max(a, b, c)
                if current_max < best_max_denom:
                    best_max_denom = current_max
                    best_solution = [a, b, c]

    if best_solution is None:
        # Fallback: Greedy (seed behavior)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return best_solution
