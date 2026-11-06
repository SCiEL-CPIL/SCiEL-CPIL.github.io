"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import log

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    target = Fraction(5, 31)
    
    best_solution = None
    best_M = 999999999
    
    def try_complete(partial, min_d):
        """Given partial denominators, compute the required last denominator.
        Returns the complete solution if valid, else None."""
        if not partial:
            return None
        remaining = target - sum(Fraction(1, d) for d in partial)
        if remaining <= 0:
            return None
        # remaining = 1/d_last, so d_last = 1/remaining
        d_last = 1 / remaining
        if d_last.denominator != 1:
            return None
        d_last_val = d_last.numerator
        if d_last_val < min_d or d_last_val > 10000000:
            return None
        if d_last_val in partial:
            return None
        return partial + [d_last_val]
    
    # Search for solutions with 2, 3, 4, 5, 6 terms
    # For n terms, we search d1 < d2 < ... < d_{n-1} and compute d_n algebraically
    # Start with small max denominator to minimize the largest
    
    # Strategy: for each term count n from 2 to 6, search systematically
    # d1 >= 7 (since 1/6 > 5/31... wait, 1/6 ≈ 0.1667, 5/31 ≈ 0.1613, so 1/6 > 5/31)
    # Actually 1/7 ≈ 0.1429 < 5/31, so d1 can be 7 or higher
    # But we could also have d1 = 6? 1/6 = 0.1667 > 0.1613 = 5/31, so 1/6 alone exceeds target
    # So d1 >= 7
    
    # For n=2: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31*31 = 961
    # Factor pairs of 961: 1*961, 31*31
    # 5a-31=1 => a=32/5 not integer
    # 5a-31=31 => a=62/5 not integer
    # 5a-31=961 => a=992/5 not integer
    # So no 2-term solution with integer denominators.
    
    # For n=3: 1/a + 1/b + 1/c = 5/31
    # Fix a, b, solve for c: 1/c = 5/31 - 1/a - 1/b
    # c = 1/(5/31 - 1/a - 1/b) must be integer > b
    
    # Search a from 7 upward, b from a+1 upward
    # Pruning: 1/a + 1/b + 1/c <= 1/a + 1/b + 1/(b+1) must be >= 5/31
    # And 1/a + 1/b < 5/31 (since 1/c > 0)
    
    max_a = 100  # reasonable upper bound for a in 3-term
    max_b = 1000  # reasonable upper bound for b
    
    for n_terms in range(2, 7):
        if time.time() > deadline:
            break
        
        if n_terms == 2:
            # No solution as shown above, skip
            continue
        elif n_terms == 3:
            # 1/a + 1/b + 1/c = 5/31, a < b < c
            for a in range(7, 100):
                if time.time() > deadline:
                    break
                # 1/b + 1/c = 5/31 - 1/a must be positive
                rem_ab = target - Fraction(1, a)
                if rem_ab <= 0:
                    continue
                # b must satisfy: 1/b < rem_ab, so b > 1/rem_ab
                b_min = int(1 / rem_ab) + 1
                if b_min <= a:
                    b_min = a + 1
                # Also 1/b + 1/c >= 1/b + 1/(b+1) > rem_ab for feasibility
                # Max possible: 1/b + 1/c <= 1/b + 1/(b+1)
                # We need 1/b + 1/c = rem_ab, so 1/b < rem_ab
                # And c = 1/(rem_ab - 1/b) must be integer > b
                for b in range(b_min, 2000):
                    if time.time() > deadline:
                        break
                    rem_c = rem_ab - Fraction(1, b)
                    if rem_c <= 0:
                        break  # b too small, but we're increasing b so rem_c decreases
                    c_val = 1 / rem_c
                    if c_val.denominator != 1:
                        continue
                    c = c_val.numerator
                    if c <= b:
                        continue
                    if c > 10000000:
                        continue
                    sol = [a, b, c]
                    if best_solution is None or c < best_M:
                        best_solution = sol
                        best_M = c
        elif n_terms == 4:
            # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
            # Fix a, b, c, solve for d
            for a in range(7, 50):
                if time.time() > deadline:
                    break
                rem_bcd = target - Fraction(1, a)
                if rem_bcd <= 0:
                    continue
                b_min = int(1 / rem_bcd) + 1
                if b_min <= a:
                    b_min = a + 1
                for b in range(b_min, 500):
                    if time.time() > deadline:
                        break
                    rem_cd = rem_bcd - Fraction(1, b)
                    if rem_cd <= 0:
                        break
                    c_min = int(1 / rem_cd) + 1
                    if c_min <= b:
                        c_min = b + 1
                    for c in range(c_min, 2000):
                        if time.time() > deadline:
                            break
                        rem_d = rem_cd - Fraction(1, c)
                        if rem_d <= 0:
                            break
                        d_val = 1 / rem_d
                        if d_val.denominator != 1:
                            continue
                        d = d_val.numerator
                        if d <= c:
                            continue
                        if d > 10000000:
                            continue
                        sol = [a, b, c, d]
                        if best_solution is None or d < best_M:
                            best_solution = sol
                            best_M = d
        elif n_terms == 5:
            # 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
            for a in range(7, 30):
                if time.time() > deadline:
                    break
                rem_bcde = target - Fraction(1, a)
                if rem_bcde <= 0:
                    continue
                b_min = int(1 / rem_bcde) + 1
                if b_min <= a:
                    b_min = a + 1
                for b in range(b_min, 200):
                    if time.time() > deadline:
                        break
                    rem_cde = rem_bcde - Fraction(1, b)
                    if rem_cde <= 0:
                        break
                    c_min = int(1 / rem_cde) + 1
                    if c_min <= b:
                        c_min = b + 1
                    for c in range(c_min, 500):
                        if time.time() > deadline:
                            break
                        rem_de = rem_cde - Fraction(1, c)
                        if rem_de <= 0:
                            break
                        d_min = int(1 / rem_de) + 1
                        if d_min <= c:
                            d_min = c + 1
                        for d in range(d_min, 1500):
                            if time.time() > deadline:
                                break
                            rem_e = rem_de - Fraction(1, d)
                            if rem_e <= 0:
                                break
                            e_val = 1 / rem_e
                            if e_val.denominator != 1:
                                continue
                            e = e_val.numerator
                            if e <= d:
                                continue
                            if e > 10000000:
                                continue
                            sol = [a, b, c, d, e]
                            if best_solution is None or e < best_M:
                                best_solution = sol
                                best_M = e
        elif n_terms == 6:
            # 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31
            for a in range(7, 20):
                if time.time() > deadline:
                    break
                rem_bcdef = target - Fraction(1, a)
                if rem_bcdef <= 0:
                    continue
                b_min = int(1 / rem_bcdef) + 1
                if b_min <= a:
                    b_min = a + 1
                for b in range(b_min, 100):
                    if time.time() > deadline:
                        break
                    rem_cdef = rem_bcdef - Fraction(1, b)
                    if rem_cdef <= 0:
                        break
                    c_min = int(1 / rem_cdef) + 1
                    if c_min <= b:
                        c_min = b + 1
                    for c in range(c_min, 300):
                        if time.time() > deadline:
                            break
                        rem_def = rem_cdef - Fraction(1, c)
                        if rem_def <= 0:
                            break
                        d_min = int(1 / rem_def) + 1
                        if d_min <= c:
                            d_min = c + 1
                        for d in range(d_min, 800):
                            if time.time() > deadline:
                                break
                            rem_ef = rem_def - Fraction(1, d)
                            if rem_ef <= 0:
                                break
                            e_min = int(1 / rem_ef) + 1
                            if e_min <= d:
                                e_min = d + 1
                            for e in range(e_min, 2000):
                                if time.time() > deadline:
                                    break
                                rem_f = rem_ef - Fraction(1, e)
                                if rem_f <= 0:
                                    break
                                f_val = 1 / rem_f
                                if f_val.denominator != 1:
                                    continue
                                f = f_val.numerator
                                if f <= e:
                                    continue
                                if f > 10000000:
                                    continue
                                sol = [a, b, c, d, e, f]
                                if best_solution is None or f < best_M:
                                    best_solution = sol
                                    best_M = f
    
    if best_solution is None:
        # Fallback: known solution
        best_solution = [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
