def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    from fractions import Fraction
    from math import isqrt, gcd
    
    target = Fraction(5, 31)
    
    best_max_den = float('inf')
    best_solution = None
    
    def try_solve_two(rem, min_den, max_allowed):
        """Solve 1/x + 1/y = rem where x >= min_den, y > x, both <= max_allowed.
        Returns (x, y) or None."""
        if rem <= 0:
            return None
        n, d = rem.numerator, rem.denominator
        # (nx - d)(ny - d) = d^2
        d2 = d * d
        limit = isqrt(d2)
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            # x = (u + d) / n, y = (v + d) / n
            bu = u + d
            bv = v + d
            if bu % n != 0 or bv % n != 0:
                continue
            x = bu // n
            y = bv // n
            if x < 1 or y < 1:
                continue
            if x > y:
                x, y = y, x
            if x < min_den:
                continue
            if x == y:
                continue
            if x > max_allowed or y > max_allowed:
                continue
            return (x, y)
        return None
    
    def try_solve_three(rem, min_den, max_allowed):
        """Solve 1/x + 1/y + 1/z = rem where x >= min_den, y > x, z > y, all <= max_allowed.
        Returns (x, y, z) or None."""
        if rem <= 0:
            return None
        # x must satisfy: 1/x < rem, so x > 1/rem
        x_min = max(min_den, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if x_min > max_allowed:
            return None
        # Also 2/x < rem (since y > x, z > y, so 1/y + 1/z < 2/x)
        # So x > 2/rem
        x_min2 = (2 * rem.denominator + rem.numerator - 1) // rem.numerator
        x_min = max(x_min, x_min2)
        
        for x in range(x_min, max_allowed + 1):
            if time.time() > DEADLINE:
                return None
            rem2 = rem - Fraction(1, x)
            if rem2 <= 0:
                continue
            result = try_solve_two(rem2, x + 1, max_allowed)
            if result is not None:
                return (x, result[0], result[1])
        return None
    
    # Strategy: enumerate first 1 or 2 denominators, then solve remaining algebraically
    
    # For 3-term: fix a, solve 1/b + 1/c = 5/31 - 1/a
    for a in range(7, 200):
        if time.time() > DEADLINE:
            break
        if a >= best_max_den:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        result = try_solve_two(rem, a + 1, best_max_den - 1 if best_max_den < float('inf') else 10000000)
        if result is not None:
            b, c = result
            sol = [a, b, c]
            m = max(sol)
            if m < best_max_den:
                # Verify
                check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                if check == target:
                    best_max_den = m
                    best_solution = sol
    
    # For 4-term: fix a, b, solve 1/c + 1/d = 5/31 - 1/a - 1/b
    for a in range(7, 100):
        if time.time() > DEADLINE:
            break
        if a >= best_max_den:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        # b must satisfy: 1/b < rem_a, and 1/b + 1/c + 1/d = rem_a with c > b, d > c
        # So 1/b < rem_a and 2/b < rem_a (roughly, since c > b means 1/c < 1/b)
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator)
        b_max = min(best_max_den - 1 if best_max_den < float('inf') else 1000, 
                    (2 * rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator)
        
        for b in range(b_min, min(b_max + 1, 500)):
            if time.time() > DEADLINE:
                break
            if b >= best_max_den:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            result = try_solve_two(rem_b, b + 1, best_max_den - 1 if best_max_den < float('inf') else 10000000)
            if result is not None:
                c, d = result
                sol = [a, b, c, d]
                m = max(sol)
                if m < best_max_den:
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d)
                    if check == target:
                        best_max_den = m
                        best_solution = sol
    
    # For 5-term: fix a, b, c, solve 1/d + 1/e = 5/31 - 1/a - 1/b - 1/c
    for a in range(7, 50):
        if time.time() > DEADLINE:
            break
        if a >= best_max_den:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator)
        for b in range(b_min, min(b_min + 50, 200)):
            if time.time() > DEADLINE:
                break
            if b >= best_max_den:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator)
            for c in range(c_min, min(c_min + 30, 300)):
                if time.time() > DEADLINE:
                    break
                if c >= best_max_den:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                result = try_solve_two(rem_c, c + 1, best_max_den - 1 if best_max_den < float('inf') else 10000000)
                if result is not None:
                    d, e = result
                    sol = [a, b, c, d, e]
                    m = max(sol)
                    if m < best_max_den:
                        check = sum(Fraction(1, x) for x in sol)
                        if check == target:
                            best_max_den = m
                            best_solution = sol
    
    if best_solution is not None and len(best_solution) <= 256:
        # Final verification
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    used = set()
    while r > 0 and len(values) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in used:
            d += 1
            while d in used:
                d += 1
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
    return values
