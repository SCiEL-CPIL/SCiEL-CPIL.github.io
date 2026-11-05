def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)

    best_max_den = 10000001
    best_solution = None

    # Phase 1: Exhaustive 3-term search
    # 1/a + 1/b + 1/c = 5/31, WLOG a <= b <= c
    # a >= 7 (since 1/a < 5/31), and a <= 18 (since 3/a >= 5/31)
    for a in range(7, 19):
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

        # We need (n*b - d)(n*c - d) = d^2
        # Actually: 1/b + 1/c = n/d => (b+c)/(bc) = n/d => d(b+c) = n*bc
        # n*bc - d*b - d*c = 0
        # (n*b - d)(n*c - d) = d^2
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

                m = max(a, b, c)
                if m < best_max_den:
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    if check == target:
                        best_max_den = m
                        best_solution = [a, b, c]

    # Phase 2: DFS for 4-term solutions with max denominator < best_max_den
    # Search up to best_max_den - 1
    def dfs4(remaining, min_den, max_bound, current, used_set):
        """Find denominators for remaining fraction, each >= min_den, <= max_bound, distinct."""
        nonlocal best_max_den, best_solution
        if time.time() - start_time > DEADLINE:
            return False
        if remaining == 0:
            m = max(current) if current else 0
            if m < best_max_den:
                best_max_den = m
                best_solution = list(current)
            return True
        if remaining < 0:
            return False
        if len(current) == 4:
            return False

        n = remaining.numerator
        d = remaining.denominator

        k = 4 - len(current)  # number of terms still to place
        # x > d/n (so 1/x < remaining)
        # x <= k * d / n (so k/x >= remaining)
        lo = max(min_den, d // n + 1)
        hi = min(max_bound, (k * d) // n)
        
        if lo > hi:
            return False

        for x in range(lo, hi + 1):
            if time.time() - start_time > DEADLINE:
                return False
            if x in used_set:
                continue
            new_rem = remaining - Fraction(1, x)
            if new_rem < 0:
                break
            if new_rem == 0:
                if len(current) + 1 <= 4:
                    m = max(current + [x])
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = list(current) + [x]
                continue
            current.append(x)
            used_set.add(x)
            dfs4(new_rem, x + 1, max_bound, current, used_set)
            current.pop()
            used_set.discard(x)
        return False

    # Try 4-term solutions with max_bound from 20 up to best_max_den-1
    upper = min(best_max_den - 1, 1000)
    for max_bound in range(20, upper + 1):
        if time.time() - start_time > DEADLINE:
            break
        dfs4(target, 7, max_bound, [], set())
        if best_solution is not None and best_max_den <= max_bound:
            break

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
