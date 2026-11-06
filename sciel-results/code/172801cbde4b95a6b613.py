"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # 1. Find best 3-term solution: 1/a + 1/b + 1/c = 5/31
    # a in [7, 18]
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        N = r.numerator
        D = r.denominator
        
        # 1/b + 1/c = N/D
        # b > D/N
        # 2/b > N/D => b < 2D/N
        b_min = max(a + 1, D // N + 1)
        b_max = (2 * D) // N
        
        for b in range(b_min, b_max + 1):
            K = b * N - D
            if K <= 0:
                continue
            g = math.gcd(K, b)
            if D % (K // g) != 0:
                continue
            c = (b * D) // K
            if c <= b:
                continue
            if len({a, b, c}) < 3:
                continue
            max_denom = max(a, b, c)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = [a, b, c]

    # 2. Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a in [7, 24] (since 4/a > 5/31 => a < 124/5 = 24.8)
    # We only care about solutions with max_denom < best_max_denom
    # If best_solution is None, best_max_denom is inf, so we search all.
    # But we know 3-term exists, so best_max_denom is finite.
    
    # Optimization: We can prune if a >= best_max_denom
    # Also, since a < b < c < d, if a >= best_max_denom, we can stop.
    
    for a in range(7, min(25, best_max_denom)):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        N1 = r1.numerator
        D1 = r1.denominator
        
        # 1/b + 1/c + 1/d = N1/D1
        # b > D1/N1
        # 3/b > N1/D1 => b < 3*D1/N1
        b_min = max(a + 1, D1 // N1 + 1)
        b_max = (3 * D1) // N1
        
        # Prune if b_min >= best_max_denom
        if b_min >= best_max_denom:
            continue
            
        for b in range(b_min, min(b_max + 1, best_max_denom)):
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            N2 = r2.numerator
            D2 = r2.denominator
            
            # 1/c + 1/d = N2/D2
            # c > D2/N2
            # 2/c > N2/D2 => c < 2*D2/N2
            c_min = max(b + 1, D2 // N2 + 1)
            c_max = (2 * D2) // N2
            
            # Prune if c_min >= best_max_denom
            if c_min >= best_max_denom:
                continue
                
            # We need to find c such that:
            # d = c * D2 / (c * N2 - D2) is an integer and d > c
            # Let K = c * N2 - D2. Then d = c * D2 / K.
            # K must divide c * D2.
            # Also d < best_max_denom => c * D2 / K < best_max_denom
            # K = c * N2 - D2
            # c * D2 < best_max_denom * (c * N2 - D2)
            # c * D2 < best_max_denom * c * N2 - best_max_denom * D2
            # c * (best_max_denom * N2 - D2) > best_max_denom * D2
            # c > (best_max_denom * D2) / (best_max_denom * N2 - D2)
            
            denom_bound = best_max_denom * N2 - D2
            if denom_bound <= 0:
                # This inequality doesn't help or is trivial, just use c_max
                pass
            else:
                c_min = max(c_min, (best_max_denom * D2) // denom_bound + 1)
                
            if c_min >= best_max_denom:
                continue
                
            for c in range(c_min, min(c_max + 1, best_max_denom)):
                K = c * N2 - D2
                if K <= 0:
                    continue
                g = math.gcd(K, c)
                if D2 % (K // g) != 0:
                    continue
                d = (c * D2) // K
                if d <= c:
                    continue
                if d >= best_max_denom:
                    continue
                if len({a, b, c, d}) < 4:
                    continue
                max_denom = max(a, b, c, d)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = [a, b, c, d]

    if best_solution:
        return best_solution

    # Fallback (should not happen)
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
