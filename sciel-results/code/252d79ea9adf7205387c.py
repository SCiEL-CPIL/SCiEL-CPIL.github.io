"""Find minimum-count Egyptian fraction with denominators in [5, 10^7]."""
import math
import time
from fractions import Fraction
from itertools import combinations


def _greedy_unity(min_d=5, max_d=10_000_000):
    """Greedy algorithm for Egyptian fraction of 1 with denominators >= min_d."""
    rem_num, rem_den = 1, 1
    used = set()
    result = []
    while rem_num > 0:
        # Minimum d such that 1/d <= rem_num/rem_den, i.e. d >= rem_den/rem_num
        d = (rem_den + rem_num - 1) // rem_num
        if d < min_d:
            d = min_d
        while d in used or d > max_d:
            d += 1
        if d > max_d:
            break
        result.append(d)
        used.add(d)
        # new remainder: rem_num/rem_den - 1/d = (rem_num*d - rem_den)/(rem_den*d)
        rem_num = rem_num * d - rem_den
        rem_den = rem_den * d
        g = math.gcd(rem_num, rem_den)
        if g > 1:
            rem_num //= g
            rem_den //= g
    return result


def _try_merge_three_two(a, b, c, min_d=5, max_d=10_000_000, timeout=2.0):
    """Try to replace 1/a + 1/b + 1/c with 1/x + 1/y (x < y, both >= min_d, distinct).
    Returns (x, y) or None."""
    t = (a * b * c) // math.gcd(a * b * c, math.gcd(a * b, a * c))
    # Actually compute sum exactly
    s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
    s_num = s.numerator
    s_den = s.denominator
    start = time.time()
    # x must satisfy 1/x < s (strictly, since we need y > x), so x > s_den/s_num
    x_min = s_den // s_num + 1
    if x_min < min_d:
        x_min = min_d
    # x < y and x != a,b,c
    # Also 1/x < s, so x > s_den/s_num
    # And 1/y = s - 1/x > 0, so 1/x < s
    # y = s_den / (s_num*x - s_den), must be integer
    # s_num*x - s_den must divide s_den
    # Let's iterate x from x_min upward, bounded by time
    # Upper bound on x: y > x means 1/y < 1/x, so s - 1/x < 1/x => s < 2/x => x < 2/s = 2*s_den/s_num
    x_max = min(2 * s_den // s_num, max_d)
    for x in range(x_min, x_max + 1):
        if x == a or x == b or x == c:
            continue
        # Check if s - 1/x is a unit fraction
        # s - 1/x = (s_num*x - s_den) / (s_den * x)
        # This equals 1/y iff s_num*x - s_den = s_den/x... no.
        # (s_num*x - s_den)/(s_den*x) = 1/y => y = s_den*x/(s_num*x - s_den)
        # So s_num*x - s_den must divide s_den*x
        num_part = s_num * x - s_den
        if num_part <= 0:
            continue
        # y = s_den * x / num_part
        if (s_den * x) % num_part != 0:
            continue
        y = (s_den * x) // num_part
        if y <= x:
            continue
        if y < min_d or y > max_d:
            continue
        if y == a or y == b or y == c:
            continue
        # Verify
        if Fraction(1, x) + Fraction(1, y) == s:
            return (x, y)
        if time.time() - start > timeout:
            break
    return None


def _try_merge_two_one(a, b, min_d=5, max_d=10_000_000):
    """Try to replace 1/a + 1/b with 1/c. Returns c or None."""
    s = Fraction(1, a) + Fraction(1, b)
    if s.denominator == 1:
        c = s.numerator
        if min_d <= c <= max_d and c != a and c != b:
            return c
    return None


def _local_search(result, min_d=5, max_d=10_000_000, deadline=None):
    """Try to reduce count by merging denominators."""
    if deadline is None:
        deadline = time.time() + 30
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        n = len(result)
        # Try merging 3 into 2
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if time.time() > deadline:
                        break
                    a, b, c = result[i], result[j], result[k]
                    merged = _try_merge_three_two(a, b, c, min_d, max_d, timeout=0.5)
                    if merged:
                        x, y = merged
                        new_result = [d for idx, d in enumerate(result) if idx not in (i, j, k)]
                        new_result.extend([x, y])
                        new_result.sort()
                        # Verify distinctness
                        if len(set(new_result)) == len(new_result):
                            s = sum(Fraction(1, d) for d in new_result)
                            if s == 1 and all(min_d <= d <= max_d for d in new_result):
                                result = new_result
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        # Try merging 2 into 1
        if not improved:
            n = len(result)
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    a, b = result[i], result[j]
                    c = _try_merge_two_one(a, b, min_d, max_d)
                    if c is not None:
                        new_result = [d for idx, d in enumerate(result) if idx not in (i, j)]
                        new_result.append(c)
                        new_result.sort()
                        if len(set(new_result)) == len(new_result):
                            s = sum(Fraction(1, d) for d in new_result)
                            if s == 1 and all(min_d <= d <= max_d for d in new_result):
                                result = new_result
                                improved = True
                                break
                if improved:
                    break
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit (evaluator kills at 90s)

    min_d = 5
    max_d = 10_000_000

    # Step 1: Greedy to get an initial valid solution
    result = _greedy_unity(min_d, max_d)

    if not result or len(result) > 256:
        raise RuntimeError("Greedy produced invalid result")

    # Verify
    s = sum(Fraction(1, d) for d in result)
    if s != 1:
        raise RuntimeError("Greedy result does not sum to 1")

    # Step 2: Local search to reduce count
    result = _local_search(result, min_d, max_d, deadline)

    # Final verification
    if len(result) < 1 or len(result) > 256:
        raise RuntimeError("Result out of bounds")
    if len(set(result)) != len(result):
        raise RuntimeError("Result has duplicates")
    if any(d < min_d or d > max_d for d in result):
        raise RuntimeError("Denominator out of range")
    s = sum(Fraction(1, d) for d in result)
    if s != 1:
        raise RuntimeError("Result does not sum to 1")

    return result
