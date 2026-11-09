"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    best_max_den = 248  # current known solution
    best_solution = [8, 31, 248]
    
    # Direct enumeration for 3 terms: 1/a + 1/b + 1/c = 5/31
    # c = 31*a*b / (5*a*b - 31*a - 31*b)
    # Need: 5ab - 31a - 31b > 0 and (5ab - 31a - 31b) divides 31ab
    # Also a < b < c
    
    def check_3_terms(max_c):
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return
        # a must be at least ceil(31/5) = 7
        # For a given a, b must be > a and 5ab - 31a - 31b > 0 => b(5a-31) > 31a => b > 31a/(5a-31)
        # Also b < c = 31ab/(5ab-31a-31b)
        for a in range(7, max_c + 1):
            if time.time() > deadline:
                return
            if 5 * a - 31 <= 0:
                continue
            # b > 31a/(5a-31)
            b_min = (31 * a) // (5 * a - 31) + 1
            b_min = max(b_min, a + 1)
            for b in range(b_min, max_c + 1):
                if time.time() > deadline:
                    return
                denom = 5 * a * b - 31 * a - 31 * b
                if denom <= 0:
                    continue
                num = 31 * a * b
                if num % denom != 0:
                    continue
                c = num // denom
                if c <= b:
                    continue
                if c > max_c:
                    continue
                # Found a valid solution
                sol = [a, b, c]
                if max(sol) < best_max_den:
                    best_max_den = max(sol)
                    best_solution = sol
                if max(sol) <= max_c:
                    return  # Can't do better with this max_c
    
    # Search for 3-term solutions with decreasing max_c
    for mc in range(247, 6, -1):
        if time.time() > deadline:
            break
        check_3_terms(mc)
        if best_max_den <= mc:
            break
    
    # Now try 4 terms with smaller max denominator
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # For given a,b,c: 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 1/(5/31 - 1/a - 1/b - 1/c)
    # We need this to be a positive integer
    
    def check_4_terms(max_d):
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return
        # a >= 7
        for a in range(7, max_d + 1):
            if time.time() > deadline:
                return
            if 5 * a - 31 <= 0:
                continue
            # For b: need 1/b < 5/31 - 1/a => b > 1/(5/31 - 1/a) = 31a/(5a-31)
            b_min = (31 * a) // (5 * a - 31) + 1
            b_min = max(b_min, a + 1)
            for b in range(b_min, max_d + 1):
                if time.time() > deadline:
                    return
                rem2 = Fraction(5, 31) - Fraction(1, a) - Fraction(1, b)
                if rem2 <= 0:
                    continue
                # c > b, and 1/c < rem2 => c > 1/rem2
                c_min = 1 // rem2 + 1 if rem2 < 1 else b + 1
                c_min = max(c_min, b + 1)
                # Also need rem2 - 1/c > 0 => c > 1/rem2
                # And 1/d = rem2 - 1/c must be positive => c > 1/rem2
                for c in range(c_min, max_d + 1):
                    if time.time() > deadline:
                        return
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    # d = 1/rem3
                    if rem3.denominator % rem3.numerator != 0:
                        continue
                    d = rem3.denominator // rem3.numerator
                    if d <= c:
                        continue
                    if d > max_d:
                        continue
                    sol = [a, b, c, d]
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
                    if max(sol) <= max_d:
                        return
    
    for md in range(best_max_den - 1, 6, -1):
        if time.time() > deadline:
            break
        check_4_terms(md)
        if best_max_den <= md:
            break
    
    # Try 5 terms with even smaller max
    def check_5_terms(max_e):
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return
        for a in range(7, max_e + 1):
            if time.time() > deadline:
                return
            if 5 * a - 31 <= 0:
                continue
            b_min = (31 * a) // (5 * a - 31) + 1
            b_min = max(b_min, a + 1)
            for b in range(b_min, max_e + 1):
                if time.time() > deadline:
                    return
                rem2 = Fraction(5, 31) - Fraction(1, a) - Fraction(1, b)
                if rem2 <= 0:
                    continue
                c_min = 1 // rem2 + 1 if rem2 < 1 else b + 1
                c_min = max(c_min, b + 1)
                for c in range(c_min, max_e + 1):
                    if time.time() > deadline:
                        return
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    d_min = 1 // rem3 + 1 if rem3 < 1 else c + 1
                    d_min = max(d_min, c + 1)
                    for d in range(d_min, max_e + 1):
                        if time.time() > deadline:
                            return
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        if rem4.denominator % rem4.numerator != 0:
                            continue
                        e = rem4.denominator // rem4.numerator
                        if e <= d:
                            continue
                        if e > max_e:
                            continue
                        sol = [a, b, c, d, e]
                        if max(sol) < best_max_den:
                            best_max_den = max(sol)
                            best_solution = sol
                        if max(sol) <= max_e:
                            return
    
    for me in range(best_max_den - 1, 6, -1):
        if time.time() > deadline:
            break
        check_5_terms(me)
        if best_max_den <= me:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
