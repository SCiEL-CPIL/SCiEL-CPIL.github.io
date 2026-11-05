def construct():
    import time
    start_time = time.time()
    DEADLINE = 85

    from math import isqrt, gcd

    def divisors(n):
        """Return all divisors of n."""
        if n <= 0:
            return []
        small = []
        large = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                small.append(i)
                if i != n // i:
                    large.append(n // i)
            i += 1
        large.reverse()
        return small + large

    def find_3term(a, max_bound):
        """Find b,c such that 1/a+1/b+1/c=5/31 with b,c distinct from a and each other, b,c<=max_bound."""
        results = []
        num = 5 * a - 31
        if num <= 0:
            return results
        den = 31 * a
        g = gcd(num, den)
        n = num // g
        d = den // g
        d2 = d * d
        for u in divisors(d2):
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
                if b > max_bound or c > max_bound:
                    continue
                # Verify
                if 31 * b * c + 31 * a * c + 31 * a * b == 5 * a * b * c:
                    results.append((a, b, c))
        return results

    def find_4term(a, b, max_bound):
        """Find c,d such that 1/a+1/b+1/c+1/d=5/31 with c,d distinct from a,b and each other."""
        results = []
        # 1/c + 1/d = 5/31 - 1/a - 1/b = (5ab - 31b - 31a)/(31ab)
        num = 5 * a * b - 31 * b - 31 * a
        if num <= 0:
            return results
        den = 31 * a * b
        g = gcd(num, den)
        n = num // g
        d = den // g
        d2 = d * d
        for u in divisors(d2):
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                c = bu // n
                dd = bv // n
                if c <= 0 or dd <= 0:
                    continue
                if c == a or c == b or c == dd:
                    continue
                if dd == a or dd == b:
                    continue
                if c > max_bound or dd > max_bound:
                    continue
                # Verify
                if (31*b*c*d_ + 31*a*c*d_ + 31*a*b*d_ + 31*a*b*c == 
                    5*a*b*c*d_) if False else True:
                    # Just verify with integer arithmetic
                    lhs = 31*b*c*dd + 31*a*c*dd + 31*a*b*dd + 31*a*b*c
                    rhs = 5*a*b*c*dd
                    if lhs == rhs:
                        results.append((a, b, c, dd))
        return results

    best_max_den = 10000001
    best_solution = None

    # Phase 1: 3-term search using divisor enumeration
    for a in range(7, 200):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        sols = find_3term(a, best_max_den - 1)
        for sol in sols:
            m = max(sol)
            if m < best_max_den:
                best_max_den = m
                best_solution = list(sol)

    # Phase 2: 4-term search - enumerate pairs (a,b) and solve for (c,d)
    if best_solution is None or best_max_den > 20:
        for a in range(7, 50):
            if time.time() - start_time > DEADLINE:
                break
            if a >= best_max_den:
                break
            for b in range(a + 1, best_max_den):
                if time.time() - start_time > DEADLINE:
                    break
                sols = find_4term(a, b, best_max_den - 1)
                for sol in sols:
                    m = max(sol)
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = list(sol)

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    from fractions import Fraction
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
