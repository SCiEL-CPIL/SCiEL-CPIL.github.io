def construct():
    import time
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)

    # We want to find distinct denominators a, b, c such that
    # 1/a + 1/b + 1/c = 5/31
    # and minimize max(a, b, c).
    #
    # Strategy:
    # 1. For 3-term: enumerate a, then solve 1/b + 1/c = 5/31 - 1/a
    #    Given r = n/d (reduced), need (nb-d)(nc-d) = d^2
    #    So we factor d^2 and check divisibility conditions.
    # 2. For 4-term: if 3-term doesn't give good enough, try 4 terms
    #    1/a + 1/b + 1/c + 1/d = 5/31
    #    This is more complex. Let's first exhaust 3-term search efficiently.
    #
    # Key insight: to minimize max denominator, we should search for small
    # max values. The optimal 3-term solution likely has all denominators < ~100.
    #
    # For 3-term: 1/a + 1/b + 1/c = 5/31
    # WLOG a <= b <= c
    # 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 (since 1/b + 1/c <= 2/b <= 2/a) => a < 93/5 = 18.6, so a <= 18
    # Wait, that's not quite right. 1/b + 1/c <= 2/a (when b,c >= a)
    # So 5/31 - 1/a <= 2/a => 5/31 <= 3/a => a <= 93/5 = 18.6
    # So a is in [7, 18] for WLOG a <= b <= c.
    #
    # But wait, we don't need WLOG ordering for the search. We just need distinct values.
    # Let me think more carefully.
    #
    # Actually, let's just enumerate a from 7 upward, and for each a, find b, c.
    # To minimize max(a,b,c), we want all three to be small.
    #
    # For a given a:
    #   r = 5/31 - 1/a = (5a - 31)/(31a)
    #   If 5a - 31 <= 0, skip (r <= 0)
    #   Reduce r = n/d
    #   Need 1/b + 1/c = n/d
    #   (nb - d)(nc - d) = d^2
    #   Let u = nb - d, v = nc - d, so u*v = d^2
    #   b = (u+d)/n, c = (v+d)/n
    #   Need u+d ≡ 0 (mod n) and v+d ≡ 0 (mod n)
    #   Need b, c positive integers, distinct from a and each other.
    
    best_max_den = 10000001
    best_solution = None

    # First, try 3-term solutions
    # a can range from 7 to some limit. Since we want to minimize max, 
    # and b,c >= a (WLOG), max is c. We want c as small as possible.
    # Let's enumerate a from 7 to 100 (should be plenty for optimal).
    
    for a in range(7, 200):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
            
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        
        g = gcd(num, den)
        n = num // g
        d = den // g
        
        d2 = d * d
        limit = isqrt(d2)
        
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            
            bu = u + d
            bv = v + d
            
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                
                if b <= 0 or c <= 0:
                    continue
                if b == a or c == a or b == c:
                    continue
                if b > 10000000 or c > 10000000:
                    continue
                
                # Check if this gives a better max
                m = max(a, b, c)
                if m < best_max_den:
                    # Verify
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    if check == target:
                        best_max_den = m
                        best_solution = [a, b, c]

    if best_solution is not None:
        return best_solution

    # If no 3-term found (unlikely), try 4-term or fallback
    # Fallback: greedy Egyptian fraction
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
