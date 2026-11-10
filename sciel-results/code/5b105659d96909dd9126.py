"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = None
    best_result = None
    
    def update_best(result):
        nonlocal best_max, best_result
        if not result:
            return
        max_d = max(result)
        if max_d > 10000000:
            return
        if len(result) < 1 or len(result) > 256:
            return
        if len(set(result)) != len(result):
            return
        if any(d < 1 for d in result):
            return
        if best_max is None or max_d < best_max:
            best_max = max_d
            best_result = list(result)
    
    # === 2-term search: 1/a + 1/b = 5/31 ===
    # 31(a+b) = 5ab => a(5b-31) = 31b => a = 31b/(5b-31)
    # Need 5b-31 > 0 => b >= 7, and a > b => b < 62/5 = 12.4
    for b in range(7, 13):
        denom = 5 * b - 31
        if denom <= 0:
            continue
        num = 31 * b
        if num % denom == 0:
            a = num // denom
            if a > b and 1 <= a <= 10000000:
                update_best([a, b])
    
    # === 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ===
    # For 3 terms, we can use the 2-term completion trick
    # 1/a + 1/b + 1/c = target
    # For each a, rem = target - 1/a, then solve 1/b + 1/c = rem
    # 1/b + 1/c = rem => (b+c)/(bc) = rem => rem*b*c = b+c => b(rem*c - 1) = c => b = c/(rem*c - 1)
    # Or: rem = (b+c)/(bc) => rem*bc - b - c = 0 => (rem*b - 1)(rem*c - 1) = 1
    # So rem*b - 1 and rem*c - 1 are divisors of 1... that's only for rem=1
    # Better: use the formula b = c / (rem*c - 1)
    # Or iterate b and check if c = 1/(rem - 1/b) is integer
    
    for a in range(7, 16):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        # Find b > a such that 1/b < rem2
        min_b = max(a + 1, -(-rem2.denominator // rem2.numerator))
        # Upper bound: rem2 < 2/b => b < 2/rem2
        max_b = min((2 * rem2.denominator) // rem2.numerator, 2000)
        if max_b < min_b:
            continue
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                break
            # c = 1/rem3 must be integer > b
            if rem3.numerator == 1:
                c = rem3.denominator
                if c > b and c <= 10000000:
                    update_best([a, b, c])
    
    # === 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d ===
    for a in range(7, 14):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        min_b = max(a + 1, -(-rem2.denominator // rem2.numerator))
        max_b = min((2 * rem2.denominator) // rem2.numerator, 800)
        if max_b < min_b:
            continue
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                break
            min_c = max(b + 1, -(-rem3.denominator // rem3.numerator))
            max_c = min((2 * rem3.denominator) // rem3.numerator, 400)
            if max_c < min_c:
                continue
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem4 = rem3 - Fraction(1, c)
                if rem4 <= 0:
                    break
                if rem4.numerator == 1:
                    d = rem4.denominator
                    if d > c and d <= 10000000:
                        update_best([a, b, c, d])
    
    # === 5-term search for potentially smaller max ===
    for a in range(7, 13):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        min_b = max(a + 1, -(-rem2.denominator // rem2.numerator))
        max_b = min((2 * rem2.denominator) // rem2.numerator, 400)
        if max_b < min_b:
            continue
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                break
            min_c = max(b + 1, -(-rem3.denominator // rem3.numerator))
            max_c = min((2 * rem3.denominator) // rem3.numerator, 250)
            if max_c < min_c:
                continue
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem4 = rem3 - Fraction(1, c)
                if rem4 <= 0:
                    break
                min_d = max(c + 1, -(-rem4.denominator // rem4.numerator))
                max_d = min((2 * rem4.denominator) // rem4.numerator, 150)
                if max_d < min_d:
                    continue
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    rem5 = rem4 - Fraction(1, d)
                    if rem5 <= 0:
                        break
                    if rem5.numerator == 1:
                        e = rem5.denominator
                        if e > d and e <= 10000000:
                            update_best([a, b, c, d, e])
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy algorithm
    r = target
    values = []
    while r > 0:
        d = -(-r.denominator // r.numerator)  # ceil(1/r)
        if d in values:
            d += 1
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
