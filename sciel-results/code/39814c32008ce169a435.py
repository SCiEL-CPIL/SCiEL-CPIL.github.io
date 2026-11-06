"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    assert (31*248 + 8*248 + 8*31) * 31 == 5 * 8 * 31 * 248, "Known solution invalid"
    
    target = Fraction(5, 31)
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    def search_2terms(p, q, min_den, max_den):
        """Find 1/b + 1/c = p/q with min_den < b < c < max_den."""
        # (p*b - q) * c = q*b => c = q*b / (p*b - q)
        # Need p*b - q > 0 => b > q/p
        # Need c > b => q*b / (p*b - q) > b => q > p*b - q => b < 2q/p
        # Need c < max_den
        results = []
        b_lo = max(min_den + 1, q // p + 1)
        b_hi = min(max_den - 1, (2 * q) // p - 1)
        if b_hi <= b_lo:
            return results
        for b in range(b_lo, b_hi + 1):
            num = p * b - q
            if num <= 0:
                continue
            if (q * b) % num != 0:
                continue
            c = (q * b) // num
            if c > b and c < max_den:
                results.append((b, c))
        return results
    
    # Try 2-term solutions
    for b, c in search_2terms(5, 31, 0, best_max):
        check_and_update([b, c])
    
    # Try 3-term solutions
    # a >= 7 (since 1/a <= 5/31 requires a >= 7), and 3/a >= 5/31 requires a <= 18
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        p, q = rem.numerator, rem.denominator
        for b, c in search_2terms(p, q, a, best_max):
            check_and_update([a, b, c])
    
    # Try 4-term solutions
    # a >= 7, and 4/a >= 5/31 requires a <= 24
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        p1, q1 = rem1.numerator, rem1.denominator
        # For b: 3/b >= rem1 => b <= 3*q1/p1, and b > q1/p1
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (3 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            p2, q2 = rem2.numerator, rem2.denominator
            for c, d in search_2terms(p2, q2, b, best_max):
                check_and_update([a, b, c, d])
    
    # Try 5-term solutions
    # a >= 7, and 5/a >= 5/31 requires a <= 30
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        p1, q1 = rem1.numerator, rem1.denominator
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (4 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            p2, q2 = rem2.numerator, rem2.denominator
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (3 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                p3, q3 = rem3.numerator, rem3.denominator
                for d, e in search_2terms(p3, q3, c, best_max):
                    check_and_update([a, b, c, d, e])
    
    # Try 6-term solutions
    # a >= 7, and 6/a >= 5/31 requires a <= 37
    for a in range(7, min(38, best_max)):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        p1, q1 = rem1.numerator, rem1.denominator
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (5 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            p2, q2 = rem2.numerator, rem2.denominator
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (4 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                p3, q3 = rem3.numerator, rem3.denominator
                d_lo = max(c + 1, q3 // p3 + 1)
                d_hi = min(best_max - 1, (3 * q3) // p3 - 1)
                if d_hi <= d_lo:
                    continue
                for d in range(d_lo, d_hi + 1):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    p4, q4 = rem4.numerator, rem4.denominator
                    for e, f in search_2terms(p4, q4, d, best_max):
                        check_and_update([a, b, c, d, e, f])
    
    # Try 7-term solutions
    # a >= 7, and 7/a >= 5/31 requires a <= 43
    for a in range(7, min(44, best_max)):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        p1, q1 = rem1.numerator, rem1.denominator
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (6 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            p2, q2 = rem2.numerator, rem2.denominator
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (5 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                p3, q3 = rem3.numerator, rem3.denominator
                d_lo = max(c + 1, q3 // p3 + 1)
                d_hi = min(best_max - 1, (4 * q3) // p3 - 1)
                if d_hi <= d_lo:
                    continue
                for d in range(d_lo, d_hi + 1):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    p4, q4 = rem4.numerator, rem4.denominator
                    e_lo = max(d + 1, q4 // p4 + 1)
                    e_hi = min(best_max - 1, (3 * q4) // p4 - 1)
                    if e_hi <= e_lo:
                        continue
                    for e in range(e_lo, e_hi + 1):
                        if time.time() > deadline:
                            break
                        rem5 = rem4 - Fraction(1, e)
                        if rem5 <= 0:
                            continue
                        p5, q5 = rem5.numerator, rem5.denominator
                        for f, g in search_2terms(p5, q5, e, best_max):
                            check_and_update([a, b, c, d, e, f, g])
    
    return best
# EVOLVE-BLOCK-END
