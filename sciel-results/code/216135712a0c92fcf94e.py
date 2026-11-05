def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)
    best_max_den = 10000001
    best_solution = None

    def check_solution(denoms):
        nonlocal best_max_den, best_solution
        if len(denoms) < 1 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        for d in denoms:
            if d < 1 or d > 10000000:
                return
        s = sum(Fraction(1, d) for d in denoms)
        if s == target:
            m = max(denoms)
            if m < best_max_den:
                best_max_den = m
                best_solution = list(denoms)

    def solve_2term(r):
        """Solve 1/b + 1/c = r where r = n/d reduced. Return list of (b,c) pairs."""
        if r <= 0:
            return []
        n = r.numerator
        d = r.denominator
        d2 = d * d
        limit = isqrt(d2)
        results = []
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                if b > 0 and c > 0:
                    # Ensure b <= c for consistency, though order doesn't matter for set
                    if b > c:
                        b, c = c, b
                    results.append((b, c))
        return results

    # Phase 1: Exhaustive 3-term search
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # 3/a > 5/31 => a < 18.6
    # 1/a < 5/31 => a > 6.2
    for a in range(7, 19):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        
        pairs = solve_2term(r)
        for b, c in pairs:
            if b != a and c != a and b != c:
                if b <= 10000000 and c <= 10000000:
                    check_solution([a, b, c])

    # Phase 2: 4-term search
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # 4/a > 5/31 => a < 24.8
    # 1/a < 5/31 => a > 6.2
    for a in range(7, 25):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        
        # b > a
        # 3/b > r1 => b < 3/r1
        # 1/b < r1 => b > 1/r1
        b_lo = max(a + 1, r1.denominator // r1.numerator + 1)
        b_hi = min(best_max_den - 1, (3 * r1.denominator) // r1.numerator)
        
        if b_lo > b_hi:
            continue
            
        for b in range(b_lo, b_hi + 1):
            if time.time() - start_time > DEADLINE:
                break
            if b >= best_max_den:
                break
            
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
                
            # c > b
            # 2/c > r2 => c < 2/r2
            # 1/c < r2 => c > 1/r2
            c_lo = max(b + 1, r2.denominator // r2.numerator + 1)
            c_hi = min(best_max_den - 1, (2 * r2.denominator) // r2.numerator)
            
            if c_lo > c_hi:
                continue
                
            for c in range(c_lo, c_hi + 1):
                if time.time() - start_time > DEADLINE:
                    break
                if c >= best_max_den:
                    break
                    
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                    
                # d > c
                # 1/d = r3 => d = 1/r3
                if r3.numerator == 1:
                    d = r3.denominator
                    if d > c and d != a and d != b and d != c:
                        if d <= 10000000:
                            check_solution([a, b, c, d])

    if best_solution is not None:
        return best_solution

    # Fallback: Greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
