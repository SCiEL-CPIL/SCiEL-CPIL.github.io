"""
Model-and-Optimize approach for Egyptian fraction of 5/31:
Goal: Minimize the LARGEST denominator in a representation with 1..256 distinct terms.

Key insight: More terms allow keeping all denominators small.
- 3-term: max ~1736
- 4-term: potentially much smaller
- 5-term: potentially even smaller

Strategy:
1. Quickly find best 3-term solution (upper bound).
2. Exhaustive 4-term search: 1/a+1/b+1/c+1/d = 5/31, a<b<c<d, d < current_best.
   For each (a,b,c), compute d = 1/(5/31 - 1/a - 1/b - 1/c) and check integrality.
3. If time permits, 5-term search similarly.

All arithmetic done with integer fractions to avoid floating point errors.
"""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    target = Fraction(5, 31)

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # --- 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ---
    # a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # 1/b + 1/c = rem1, b > a
        # b < 1/rem1 (since 1/b < rem1)
        # b > a, and 2/b >= rem1 => b <= 2/rem1
        max_b = int(2 / rem1)
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # 1/c = rem2 => c = 1/rem2
            c_frac = 1 / rem2
            c = c_frac.numerator // c_frac.denominator
            if c_frac.denominator != 1:
                continue
            if c <= b:
                continue
            update_best([a, b, c])

    if best_result is not None:
        # Now try 4-term search to potentially improve
        pass

    # --- 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d < best_max ---
    # a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue

        # b range: b > a, 3/b >= rem1 => b <= 3/rem1
        max_b = int(3 / rem1)
        min_b = a + 1

        for b in range(min_b, max_b + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue

            # c range: c > b, 2/c >= rem2 => c <= 2/rem2
            # Also c < best_max (since d > c and d < best_max)
            max_c = min(int(2 / rem2), best_max - 1)
            min_c = b + 1

            if max_c < min_c:
                continue

            for c in range(min_c, max_c + 1):
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                # 1/d = rem3 => d = 1/rem3
                d_frac = 1 / rem3
                d = d_frac.numerator // d_frac.denominator
                if d_frac.denominator != 1:
                    continue
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                update_best([a, b, c, d])

    if best_result is not None:
        # Try 5-term search if we have time
        pass

    # --- 5-term search: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ---
    # Only if best_max is still large enough to be worth searching
    if best_max < float('inf'):
        for a in range(7, 19):
            if time.time() > deadline:
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue

            max_b = int(4 / rem1)
            min_b = a + 1

            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue

                max_c = min(int(3 / rem2), best_max - 1)
                min_c = b + 1

                if max_c < min_c:
                    continue

                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue

                    max_d = min(int(2 / rem3), best_max - 1)
                    min_d = c + 1

                    if max_d < min_d:
                        continue

                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        # 1/e = rem4 => e = 1/rem4
                        e_frac = 1 / rem4
                        e = e_frac.numerator // e_frac.denominator
                        if e_frac.denominator != 1:
                            continue
                        if e <= d:
                            continue
                        if e >= best_max:
                            continue
                        update_best([a, b, c, d, e])

    if best_result is not None:
        return best_result

    # Final fallback
    return [7, 8, 9, 10000000]
