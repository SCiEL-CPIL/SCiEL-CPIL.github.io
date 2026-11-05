def construct():
    from fractions import Fraction
    import time
    from math import gcd
    
    deadline = time.time() + 85  # 85s deadline with 5s margin
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max = 10000001
    
    # For n-term Egyptian fraction of 5/31:
    # 1/a1 + 1/a2 + ... + 1/an = 5/31
    # Key: all denominators must be >= 7 (since 1/6 = 0.1667 > 5/31 ≈ 0.1613)
    # 
    # Strategy: Use the identity that for 1/a + 1/b = p/q:
    #   (a - q/p)(b - q/p) = (q/p)^2  [when p|q]
    # More generally, for 1/a + 1/b = p/q:
    #   ab/(a+b) = q/p => p*ab = q*(a+b) => p*ab - q*a - q*b = 0
    #   (pa - q)(pb - q) = q^2
    #
    # For 5/31: (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 31^2, divisors: 1, 31, 961
    # So possible (5a-31, 5b-31) pairs: (1,961), (31,31), (961,1)
    # 5a-31=1 => a=32/5 not integer
    # 5a-31=31 => a=62/5 not integer  
    # 5a-31=961 => a=992/5 not integer
    # No 2-term solution.
    #
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # Fix a, then 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # Let p = 5a-31, q = 31a
    # (pb - q)(pc - q) = q^2
    # We need p | q^2 and the factors to give integer b,c
    #
    # More efficient: iterate a, compute r = 5/31 - 1/a, then solve 1/b + 1/c = r
    
    def solve_two_terms(r_num, r_den):
        """Find all pairs (b,c) with b<c, b>=7, 1/b+1/c = r_num/r_den (reduced)"""
        # 1/b + 1/c = r_num/r_den
        # r_num*b*c = r_den*(b+c)
        # r_num*b*c - r_den*b - r_den*c = 0
        # (r_num*b - r_den)(r_num*c - r_den) = r_den^2
        # Let x = r_num*b - r_den, y = r_num*c - r_den
        # x*y = r_den^2, x < y (since b < c)
        # b = (x + r_den)/r_num, c = (y + r_den)/r_num
        # Need x+y... actually x*y = r_den^2
        # For each divisor d of r_den^2 with d < r_den:
        #   x = d, y = r_den^2/d
        #   b = (d + r_den)/r_num, c = (r_den^2/d + r_den)/r_num
        #   Check b, c are positive integers with b < c and b >= 7
        
        solutions = []
        rden_sq = r_den * r_den
        
        # Iterate over divisors of rden_sq up to sqrt
        limit = int(rden_sq ** 0.5)
        for d in range(1, limit + 1):
            if rden_sq % d != 0:
                continue
            x = d
            y = rden_sq // d
            # b = (x + r_den) / r_num
            if (x + r_den) % r_num != 0:
                continue
            b = (x + r_den) // r_num
            if (y + r_den) % r_num != 0:
                continue
            c = (y + r_den) // r_num
            if b < 7 or c < 7:
                continue
            if b >= c:
                continue
            solutions.append((b, c))
        return solutions
    
    # Search 3-term solutions
    # a must satisfy: 5a - 31 > 0 => a >= 7
    # Also need r = (5a-31)/(31a) to be expressible as sum of two unit fractions with denominators >= 7
    # r < 5/31, and 1/7 + 1/8 = 15/56 ≈ 0.268 > 5/31, so we can have small b,c
    # But we need b,c >= a (for distinctness and ordering)
    
    found_any = False
    for a in range(7, 500):
        if time.time() > deadline:
            break
        r_num = 5 * a - 31
        r_den = 31 * a
        if r_num <= 0:
            continue
        # Reduce fraction
        g = gcd(r_num, r_den)
        r_num_r = r_num // g
        r_den_r = r_den // g
        
        # Need r_num_r * b * c = r_den_r * (b+c)
        # (r_num_r * b - r_den_r)(r_num_r * c - r_den_r) = r_den_r^2
        # For efficiency, we need r_den_r^2 to not be too large
        # r_den_r = 31a/g, so r_den_r^2 = (31a/g)^2
        # For a=7: r_num=4, r_den=217, g=gcd(4,217)=1, r_den_r=217, r_den_r^2=47089
        # For a=10: r_num=19, r_den=310, g=1, r_den_r^2=96100
        # This grows as a^2, so limit a
        
        # Only proceed if r_den_r^2 is manageable
        if r_den_r > 2000:
            continue
            
        pairs = solve_two_terms(r_num_r, r_den_r)
        for (b, c) in pairs:
            if b > a and c > b:
                solution = [a, b, c]
                max_val = max(solution)
                if max_val < best_max:
                    best_max = max_val
                    best_solution = solution
                    found_any = True
    
    if best_solution:
        return best_solution
    
    # Try 4-term solutions with smaller max
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix a, b, then solve 1/c + 1/d = 5/31 - 1/a - 1/b
    found_4term = False
    for a in range(7, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if time.time() > deadline:
                break
            # r = 5/31 - 1/a - 1/b = (5ab - 31b - 31a)/(31ab)
            r_num = 5 * a * b - 31 * b - 31 * a
            r_den = 31 * a * b
            if r_num <= 0:
                continue
            g = gcd(r_num, r_den)
            r_num_r = r_num // g
            r_den_r = r_den // g
            if r_den_r > 5000:
                continue
            
            pairs = solve_two_terms(r_num_r, r_den_r)
            for (c, d) in pairs:
                if c > b and d > c:
                    solution = [a, b, c, d]
                    max_val = max(solution)
                    if max_val < best_max:
                        best_max = max_val
                        best_solution = solution
                        found_4term = True
    
    if best_solution:
        return best_solution
    
    # Fallback: greedy
    r, values = Fraction(5, 31), []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
