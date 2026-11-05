def construct():
    import time
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    target = Fraction(5, 31)

    # Verify the known optimal 3-term solution immediately
    # 1/8 + 1/31 + 1/248 = ?
    # = 31/248 + 8/248 + 1/248 = 40/248 = 5/31 ✓
    known_solution = [8, 31, 248]
    if sum(Fraction(1, d) for d in known_solution) == target:
        best_solution = known_solution
        best_max = 248
    else:
        best_solution = None
        best_max = float('inf')

    # Factorization helpers
    def factorize(n):
        if n <= 1:
            return {}
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp):
                    val *= p
                    new_divs.append(val)
            divs += new_divs
        return divs

    # Find 2-term solution (c, d) with min_den < c < d <= max_den_limit
    # such that 1/c + 1/d = rem. Returns pair with smallest d, or None.
    def find_2_term_min_d(rem, min_den, max_den_limit):
        num, den = rem.numerator, rem.denominator
        D_sq = den * den

        fac_den = factorize(den)
        fac_D_sq = {p: 2 * e for p, e in fac_den.items()}
        divs = get_divisors(fac_D_sq)

        lower_x = num * min_den - den
        upper_y = num * max_den_limit - den

        best_d = float('inf')
        best_pair = None

        for x in divs:
            if x >= den:
                continue
            if x <= lower_x:
                continue

            y = D_sq // x
            if y <= x:
                continue
            if y > upper_y:
                continue

            if (x + den) % num != 0:
                continue
            c = (x + den) // num
            if c <= min_den:
                continue

            if (y + den) % num != 0:
                continue
            d = (y + den) // num
            if d <= c:
                continue
            if d > max_den_limit:
                continue

            if d < best_d:
                best_d = d
                best_pair = (c, d)

        return best_pair

    # Search for 4-term solutions with max denominator < best_max
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d < best_max
    if best_solution is not None and time.time() < deadline - 5.0:
        limit = best_max - 1  # We need d < best_max

        # a: 4/a >= 5/31 => a <= 124/5 = 24.8, and 1/a < 5/31 => a >= 7
        for a in range(7, 25):
            if time.time() > deadline - 3.0:
                break

            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue

            # b: 3/b >= rem1 => b <= 3/rem1, and b > a
            max_b = int(3 / rem1)
            if max_b < a + 1:
                continue

            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 2.0:
                    break

                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue

                # c: 2/c >= rem2 => c <= 2/rem2, and c > b
                max_c = int(2 / rem2)
                if max_c < b + 1:
                    continue

                # For each c, we need 1/d = rem2 - 1/c
                # d = 1/(rem2 - 1/c) must be integer, d > c, d <= limit
                # This is a 1-term solve, which is faster than 2-term divisor scan
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline - 1.0:
                        break

                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue

                    # d = 1/rem3 must be integer
                    if rem3.denominator % rem3.numerator == 0:
                        d = rem3.denominator // rem3.numerator
                        if d > c and d <= limit:
                            sol = [a, b, c, d]
                            if sum(Fraction(1, dd) for dd in sol) == target:
                                if d < best_max:
                                    best_max = d
                                    best_solution = sol

    # If we have time, try 5-term solutions
    if best_solution is not None and best_max > 100 and time.time() < deadline - 10.0:
        limit = best_max - 1

        # a: 5/a >= 5/31 => a <= 31, 1/a < 5/31 => a >= 7
        for a in range(7, 32):
            if time.time() > deadline - 5.0:
                break

            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue

            max_b = int(4 / rem1)
            if max_b < a + 1:
                continue

            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break

                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue

                max_c = int(3 / rem2)
                if max_c < b + 1:
                    continue

                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline - 2.0:
                        break

                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue

                    max_d = int(2 / rem3)
                    if max_d < c + 1:
                        continue

                    for d in range(c + 1, max_d + 1):
                        if time.time() > deadline - 1.0:
                            break

                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue

                        if rem4.denominator % rem4.numerator == 0:
                            e = rem4.denominator // rem4.numerator
                            if e > d and e <= limit:
                                sol = [a, b, c, d, e]
                                if sum(Fraction(1, dd) for dd in sol) == target:
                                    if e < best_max:
                                        best_max = e
                                        best_solution = sol

    # Fallback
    if best_solution is None:
        best_solution = [8, 31, 248]
        best_max = 248

    return best_solution
