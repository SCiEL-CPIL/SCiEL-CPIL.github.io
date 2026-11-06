"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    best_solution = None
    best_max = 10000001
    
    # We search for 3-term expansions 1/a + 1/b + 1/c = 5/31
    # with a < b < c.
    # 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # Let P = 5a-31, Q = 31a.
    # (P*b - Q)(P*c - Q) = Q^2
    # Iterate a.
    # a must be > 31/5 = 6.2, so a >= 7.
    # If a is large, 1/a is small, so 1/b+1/c is close to 5/31.
    # b > a, c > b.
    # Max denominator is c. We want to minimize c.
    
    for a in range(7, 100):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
            
        q2 = q * q
        
        # Find divisors of q2
        # d = p*b - q  => b = (d+q)/p
        # e = p*c - q  => c = (e+q)/p
        # d*e = q2
        # We need b > a and c > b.
        
        # Iterate d up to sqrt(q2)
        limit = int(math.isqrt(q2))
        for d in range(1, limit + 1):
            if q2 % d != 0:
                continue
            
            e = q2 // d
            
            # Check if b is integer and > a
            if (d + q) % p != 0:
                continue
            b = (d + q) // p
            if b <= a:
                continue
            
            # Check if c is integer and > b
            if (e + q) % p != 0:
                continue
            c = (e + q) // p
            if c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                
            # Also check the case where d and e are swapped?
            # If we swap d and e, b and c swap.
            # Since we require b < c, and we iterate d <= e,
            # the first valid pair (b,c) will have b < c if d < e?
            # d = p*b - q. Since b < c, d < e.
            # So iterating d up to sqrt covers all unique pairs {b,c}.
            
    if best_solution is not None:
        return best_solution
    
    return [8, 31, 248]
# EVOLVE-BLOCK-END
