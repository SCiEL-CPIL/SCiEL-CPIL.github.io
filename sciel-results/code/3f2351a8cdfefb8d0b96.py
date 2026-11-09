def construct():
    from fractions import Fraction
    import math
    import time

    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = [8, 31, 248] # Known good solution

    # 1. Search for 3-term solutions
    # a < b < c
    # 1/a + 1/b + 1/c = 5/31
    # a in [7, 18]
    for a in range(7, 19):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        num1, den1 = r1.numerator, r1.denominator
        # (num1 * b - den1) * (num1 * c - den1) = den1^2
        den1_sq = den1 * den1
        # b >= a+1 => num1*b - den1 >= num1*(a+1) - den1
        min_X = max(1, num1 * (a + 1) - den1)
        for X in range(min_X, int(math.isqrt(den1_sq)) + 1):
            if den1_sq % X != 0:
                continue
            Y = den1_sq // X
            if (X + den1) % num1 != 0:
                continue
            b = (X + den1) // num1
            if (Y + den1) % num1 != 0:
                continue
            c = (Y + den1) // num1
            if b < a + 1 or c <= b:
                continue
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]

    # 2. Search for 4-term solutions with max < best_max
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d < best_max
    limit = best_max
    if limit > 100: # Only search if we have room to improve
        for a in range(7, 25):
            if 1/a >= target:
                break
            for b in range(a + 1, limit):
                r1 = target - Fraction(1, a) - Fraction(1, b)
                if r1 <= 0:
                    break # b increasing makes r1 larger? No, 1/b decreasing, so -1/b increasing, so r1 increasing.
                    # Wait. r1 = T - 1/a - 1/b.
                    # If b increases, 1/b decreases, so -1/b increases, so r1 increases.
                    # So if r1 <= 0, it might become > 0 later?
                    # No, if r1 <= 0, we need larger 1/b, so smaller b.
                    # But b is increasing. So if r1 <= 0, it will stay <= 0 or become more positive?
                    # Let's check.
                    # b=10, 1/b=0.1. b=11, 1/b=0.09.
                    # r1 = T - 1/a - 0.1.
                    # r1' = T - 1/a - 0.09.
                    # r1' > r1.
                    # So if r1 <= 0, r1' might be > 0.
                    # So we cannot break. We must continue.
                    continue
                
                # 1/c + 1/d = r1
                # c < d < limit
                # 1/c < r1 => c > 1/r1
                # 2/c > r1 => c < 2/r1
                c_min = max(b + 1, int(1 / r1) + 1)
                c_max = min(limit - 1, int(2 / r1))
                
                if c_min > c_max:
                    continue
                
                for c in range(c_min, c_max + 1):
                    r2 = r1 - Fraction(1, c)
                    if r2 <= 0:
                        continue
                    d_num, d_den = 1, 1
                    # d = 1 / r2
                    if r2.numerator == 0:
                        continue
                    if r2.denominator % r2.numerator != 0:
                        continue
                    d = r2.denominator // r2.numerator
                    if d <= c or d >= limit:
                        continue
                    # Check distinct and sum
                    if a != b and b != c and c != d and a != c and a != d and b != d:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                            if d < best_max:
                                best_max = d
                                best_solution = [a, b, c, d]
                                limit = best_max

    return best_solution
