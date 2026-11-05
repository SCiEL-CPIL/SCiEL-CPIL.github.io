def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    # Try all possible first denominators a
    # 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 1/a > 5/31 - 2/(a+1) for feasibility, but let's just bound reasonably
    # For 3 terms: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # c > 31/5 - ... let's just search
    
    for a in range(7, 100):  # reasonable upper bound for first term
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        
        # Now find 1/b + 1/c = r1 with a < b < c
        # (r1.den * b - r1.num * c) ... 
        # 1/b + 1/c = r1 => c(r1.den*b - r1.num) = r1.den*b ... 
        # Actually: r1 = p/q in lowest terms
        # 1/b + 1/c = p/q => q(b+c) = pbc => pbc - qb - qc = 0
        # (pb - q)(pc - q) = q^2
        p = r1.numerator
        q = r1.denominator
        q2 = q * q
        
        # Find factor pairs (d, q2/d) of q2 where d = pb - q
        # So pb = d + q => b = (d+q)/p must be integer > a
        # And pc = (q2/d) + q => c = (q2/d + q)/p must be integer > b
        
        # Enumerate divisors d of q2
        # For efficiency, iterate d up to sqrt(q2)
        import math
        limit = int(math.isqrt(q2))
        for d in range(1, limit + 1):
            if q2 % d != 0:
                continue
            d1 = d
            d2 = q2 // d
            
            for d_val, d_other in [(d1, d2), (d2, d1)]:
                num_b = d_val + q
                if num_b % p != 0:
                    continue
                b = num_b // p
                if b <= a:
                    continue
                num_c = d_other + q
                if num_c % p != 0:
                    continue
                c = num_c // p
                if c <= b:
                    continue
                # Check: 1/b + 1/c == r1
                if Fraction(1, b) + Fraction(1, c) == r1:
                    max_den = c
                    if max_den < best_max:
                        best_max = max_den
                        best_solution = [a, b, c]
        
        if best_solution is not None and best_max < a * 2:  # early termination heuristic
            break
    
    # Also try 2-term and 4+ term solutions? 
    # 2-term: we showed no solution
    # Let's also try more terms if needed, but 3 terms gave 248 which is good
    
    if best_solution is None:
        # Fallback to greedy
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
