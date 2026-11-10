"""Model-and-optimize: search for Egyptian fraction of 5/31 minimizing max denominator."""
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    target = Fraction(5, 31)

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # 3-term search
    # a >= 7 (since 1/a < 5/31 requires a > 6.2, and 3/a >= 5/31 gives a <= 18.6)
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        # 1/b + 1/c = rem => (b+c)/(bc) = rem => rem*bc = b+c
        # b > 1/rem, b <= 2/rem (since 1/b < rem < 2/b when c > b)
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 2 / rem
        max_b = max_b_frac.numerator // max_b_frac.denominator

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1/rem2
            c_frac = 1 / rem2
            if c_frac.denominator == 1:
                c = c_frac.numerator
                if c > b and c <= 10000000:
                    update_best([a, b, c])

    # 4-term search
    for a in range(7, 126):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 3 / rem
        max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            min_c_frac = 1 / rem2
            min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
            max_c_frac = 2 / rem2
            max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                d_frac = 1 / rem3
                if d_frac.denominator == 1:
                    d = d_frac.numerator
                    if d > c and d <= 10000000 and d < best_max:
                        update_best([a, b, c, d])

    # 5-term search
    for a in range(7, 32):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 4 / rem
        max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            min_c_frac = 1 / rem2
            min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
            max_c_frac = 3 / rem2
            max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                min_d_frac = 1 / rem3
                min_d = max(c + 1, min_d_frac.numerator // min_d_frac.denominator + 1)
                max_d_frac = 2 / rem3
                max_d = min(max_d_frac.numerator // max_d_frac.denominator, best_max - 1)

                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    if d >= best_max:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    e_frac = 1 / rem4
                    if e_frac.denominator == 1:
                        e = e_frac.numerator
                        if e > d and e <= 10000000 and e < best_max:
                            update_best([a, b, c, d, e])

    # 6-term search (only if best_max is still large)
    if best_max > 200:
        for a in range(7, 32):
            if time.time() > deadline:
                break
            if a >= best_max:
                break
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            min_b_frac = 1 / rem
            min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
            max_b_frac = 5 / rem
            max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                if b >= best_max:
                    break
                rem2 = rem - Fraction(1, b)
                if rem2 <= 0:
                    continue
                min_c_frac = 1 / rem2
                min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
                max_c_frac = 4 / rem2
                max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    if c >= best_max:
                        break
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    min_d_frac = 1 / rem3
                    min_d = max(c + 1, min_d_frac.numerator // min_d_frac.denominator + 1)
                    max_d_frac = 3 / rem3
                    max_d = min(max_d_frac.numerator // max_d_frac.denominator, best_max - 1)

                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        if d >= best_max:
                            break
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        min_e_frac = 1 / rem4
                        min_e = max(d + 1, min_e_frac.numerator // min_e_frac.denominator + 1)
                        max_e_frac = 2 / rem4
                        max_e = min(max_e_frac.numerator // max_e_frac.denominator, best_max - 1)

                        for e in range(min_e, max_e + 1):
                            if time.time() > deadline:
                                break
                            if e >= best_max:
                                break
                            rem5 = rem4 - Fraction(1, e)
                            if rem5 <= 0:
                                continue
                            f_frac = 1 / rem5
                            if f_frac.denominator == 1:
                                f = f_frac.numerator
                                if f > e and f <= 10000000 and f < best_max:
                                    update_best([a, b, c, d, e, f])

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]
