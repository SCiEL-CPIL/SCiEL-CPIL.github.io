"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    def solve_2(rem, max_d, excl):
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        for x in range(lo, max_d + 1):
            if x in excl:
                continue
            k = p * x - q
            if k <= 0:
                continue
            if (q * x) % k:
                continue
            y = (q * x) // k
            if y <= x or y > max_d or y in excl:
                continue
            return (x, y)
        return None
    
    def solve_k(rem, max_d, excl, k, dl):
        if k == 0:
            return [] if rem == 0 else None
        if k == 1:
            if rem.denominator == 1:
                x = rem.numerator
                if 1 <= x <= max_d and x not in excl:
                    return [x]
            return None
        if k == 2:
            r = solve_2(rem, max_d, excl)
            return list(r) if r else None
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = min(k * q // p, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            if time.time() > dl:
                return None
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = solve_k(r2, max_d, excl | {x}, k - 1, dl)
            if r is not None:
                return [x] + r
        return None
    
    for C in range(7, 1000):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, C)
        if rem <= 0:
            continue
        for nt in range(1, 10):
            if time.time() > deadline:
                break
            sols = solve_k(rem, C - 1, frozenset(), nt, deadline)
            if sols is not None:
                all_d = [C] + sols
                if len(set(all_d)) == len(all_d):
                    s = sum(Fraction(1, d) for d in all_d)
                    if s == target:
                        return sorted(all_d)
    
    return [8, 31, 248]
# EVOLVE-BLOCK-END
