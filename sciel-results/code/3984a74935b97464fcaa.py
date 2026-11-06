"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start = time.time()
    deadline = start + 50

    def search(k, min_den, target, current):
        if time.time() > deadline:
            return None
        if k == 0:
            return list(current) if target == 0 else None
        if k == 1:
            if target > 0 and target.denominator == 1:
                d = target.numerator
                if d >= min_den:
                    return list(current) + [d]
            return None
        if target <= 0:
            return None
        if target > Fraction(k, min_den):
            return None
        d_max = k / target
        if d_max.denominator == 0:
            return None
        upper = d_max.numerator // d_max.denominator
        if upper < min_den:
            return None
        for d in range(min_den, upper + 1):
            if time.time() > deadline:
                return None
            rem = target - Fraction(1, d)
            if rem < 0:
                continue
            if rem > Fraction(k - 1, d + 1):
                continue
            result = search(k - 1, d + 1, rem, current + [d])
            if result is not None:
                return result
        return None

    for k in range(5, 10):
        if time.time() > deadline:
            break
        result = search(k, 5, Fraction(1), [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(d >= 5 for d in result):
                return result

    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
# EVOLVE-BLOCK-END
