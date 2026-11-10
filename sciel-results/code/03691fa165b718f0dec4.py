"""Direct approach: find Egyptian fraction of 5/31 minimizing max denominator."""
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s kill)

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
    # a >= 7 (since 1/a < 5/31 => a > 6.2), a <= 18 (since 3/a >= 5/31)
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        # 1/b + 1/c = rem, b > a
        # b > 1/rem, b <= 2/rem
        min_b = max(a + 1, int(1 / rem) + 1)
        max_b = int(2 / rem)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            c = 1 / rem2
            if c == int(c) and int(c) > b and int(c) <= 10000000:
                update_best([a, b, int(c)])

    # --- 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d ---
    # a >= 7, and 4/a >= 5/31 => a <= 125.2 => a <= 125
    # But we want small max, so let's be smart: if current best_max is M,
    # we only need to explore where d < M
    for a in range(7, 126):
        if time.time() > deadline:
            break
        if a >= best_max:
            break  # can't improve
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        # 1/b + 1/c + 1/d = rem, b > a
        # b > 1/rem, b <= 3/rem
        min_b = max(a + 1, int(1 / rem) + 1)
        max_b = min(int(3 / rem), best_max - 1)  # d > c > b, so b < best_max
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            # 1/c + 1/d = rem2, c > b
            min_c = max(b + 1, int(1 / rem2) + 1)
            max_c = min(int(2 / rem2), best_max - 1)
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                d = 1 / rem3
                if d == int(d) and int(d) > c and int(d) <= 10000000 and int(d) < best_max:
                    update_best([a, b, c, int(d)])

    # --- 5-term search: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ---
    # a >= 7, 5/a >= 5/31 => a <= 31
    for a in range(7, 32):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b = max(a + 1, int(1 / rem) + 1)
        max_b = min(int(4 / rem), best_max - 1)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            min_c = max(b + 1, int(1 / rem2) + 1)
            max_c = min(int(3 / rem2), best_max - 1)
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                min_d = max(c + 1, int(1 / rem3) + 1)
                max_d = min(int(2 / rem3), best_max - 1)
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    e = 1 / rem4
                    if e == int(e) and int(e) > d and int(e) <= 10000000 and int(e) < best_max:
                        update_best([a, b, c, d, int(e)])

    if best_result is not None:
        return best_result

    # Fallback (should never reach)
    return [7, 8, 9, 10000000]

# EVOLVE-BLOCK-END
