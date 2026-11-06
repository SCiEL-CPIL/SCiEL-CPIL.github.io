"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max_denom = float('inf')
    
    # Try 2 terms: 1/a + 1/b = 5/31
    # (5a-31)(5b-31) = 961
    # Already shown no solution.
    
    # Try 3 terms: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a, solve 1/b + 1/c = 5/31 - 1/a
    # p = 5a - 31, q = 31a
    # Need p > 0, so a >= 7 (since 5*6-31 = -1 < 0, 5*7-31 = 4 > 0)
    
    for a in range(7, 100):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        
        # (pb - q)(pc - q) = q^2
        q2 = q * q
        # Find divisors u of q2 with u <= sqrt(q2)
        # b = (u + q) / p, c = (q2//u + q) / p
        # Need b > a, c > b, both integers
        
        sqrt_q2 = int(math.isqrt(q2))
        for u in range(1, sqrt_q2 + 1):
            if q2 % u != 0:
                continue
            v = q2 // u
            # u <= v, so b <= c
            # b = (u + q) / p, c = (v + q) / p
            num_b = u + q
            num_c = v + q
            if num_b % p != 0 or num_c % p != 0:
                continue
            b = num_b // p
            c = num_c // p
            if b <= a or c <= b:
                continue
            if b < 1 or c < 1:
                continue
            # Check distinctness: a < b < c already ensures distinct
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                max_d = c
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_solution = [a, b, c]
    
    # Try 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # This is more complex. Let's try a small range.
    # For a, b with a < b, solve 1/c + 1/d = 5/31 - 1/a - 1/b
    for a in range(7, 30):
        for b in range(a + 1, 100):
            rem = target - Fraction(1, a) - Fraction(1, b)
            if rem <= 0:
                break
            # Solve 1/c + 1/d = rem, c > b, c < d
            # rem = p/q in lowest terms
            p = rem.numerator
            q = rem.denominator
            if p <= 0:
                continue
            # (pc - q)(pd - q) = q^2
            q2 = q * q
            sqrt_q2 = int(math.isqrt(q2))
            for u in range(1, sqrt_q2 + 1):
                if q2 % u != 0:
                    continue
                v = q2 // u
                num_c = u + q
                num_d = v + q
                if num_c % p != 0 or num_d % p != 0:
                    continue
                c = num_c // p
                d = num_d // p
                if c <= b or d <= c:
                    continue
                if c < 1 or d < 1:
                    continue
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                    max_d = d
                    if max_d < best_max_denom:
                        best_max_denom = max_d
                        best_solution = [a, b, c, d]
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
