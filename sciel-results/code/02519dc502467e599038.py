"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # Assume a < b < c
    # a must be at least ceil(31/5) = 7
    # We can stop when a >= best_max_denom because c > b > a, so max(c) > a
    
    for a in range(7, 1000): # Upper bound heuristic
        if a >= best_max_denom:
            break
            
        # 1/b + 1/c = target - 1/a
        r = target - Fraction(1, a)
        if r <= 0:
            continue
            
        N = r.numerator
        D = r.denominator
        
        # We need (Nb - D)(Nc - D) = D^2
        # Let X = Nb - D, Y = Nc - D
        # X * Y = D^2
        # b = (X + D) / N
        # c = (Y + D) / N
        # b, c must be integers
        # X < D (since b < c)
        
        # Iterate divisors of D^2
        D_sq = D * D
        
        # Get divisors of D_sq
        divisors = []
        for i in range(1, int(math.isqrt(D_sq)) + 1):
            if D_sq % i == 0:
                divisors.append(i)
                if i != D_sq // i:
                    divisors.append(D_sq // i)
        
        for X in divisors:
            if X >= D:
                continue
            # Check if b is integer
            if (X + D) % N != 0:
                continue
            b = (X + D) // N
            if b <= a:
                continue
            # Check if c is integer
            Y = D_sq // X
            if (Y + D) % N != 0:
                continue
            c = (Y + D) // N
            if c <= b:
                continue
            
            # Check distinctness (a, b, c are distinct by construction a < b < c)
            # Check sum (guaranteed by derivation)
            
            current_max = c
            if current_max < best_max_denom:
                best_max_denom = current_max
                best_solution = [a, b, c]
                
    if best_solution:
        return best_solution
    else:
        # Fallback to seed logic if no 3-term solution found (should not happen)
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
