"""Rewritten program: greedy Egyptian fraction with deadline guard."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Greedy Egyptian fraction: repeatedly take the largest unit fraction
    # (smallest denominator >= 5) that does not overshoot the remaining sum,
    # ensuring distinct denominators and that the remainder is still representable.
    #
    # We track the remaining fraction as a pair (num, den) in lowest terms.
    # At each step, we need the smallest integer d >= 5 such that 1/d <= num/den,
    # i.e., d >= den/num. Also d must not have been used before.
    #
    # After choosing d, the remainder becomes (num*den - den/d) / (den * d) ... 
    # Actually: num/den - 1/d = (num*d - den) / (den*d)

    def reduce(n, d):
        if n == 0:
            return (0, 1)
        # Euclidean GCD
        a, b = abs(n), d
        while b:
            a, b = b, a % b
        g = a
        return (n // g, d // g)

    num, den = 1, 1  # remaining = 1/1
    used = set()
    result = []

    while num > 0:
        if time.time() > deadline:
            # Return best found so far if it sums to 1; otherwise raise
            # But greedy is deterministic and fast; this guard is just safety.
            break

        # Find smallest d >= 5, d not in used, such that 1/d <= num/den
        # i.e., d >= ceil(den / num)
        min_d = (den + num - 1) // num  # ceil(den/num)
        if min_d < 5:
            min_d = 5

        d = min_d
        while d in used:
            d += 1

        # Check: 1/d <= num/den  <=>  den <= num * d
        if num * d < den:
            # This shouldn't happen given min_d, but guard
            d += 1
            while d in used or num * d < den:
                d += 1

        # Choose d
        result.append(d)
        used.add(d)

        # Update remainder: num/den - 1/d = (num*d - den) / (den*d)
        new_num = num * d - den
        new_den = den * d
        num, den = reduce(new_num, new_den)

        # Safety: ensure denominators stay within [5, 10000000]
        if d > 10000000:
            raise RuntimeError("denominator exceeds limit")

        # If we have too many terms, we should stop (but greedy on 1 should be fast)
        if len(result) > 256:
            raise RuntimeError("too many terms")

    # Verify the result
    if len(result) < 1 or len(result) > 256:
        raise RuntimeError("invalid term count")
    for d in result:
        if d < 5 or d > 10000000:
            raise RuntimeError("denominator out of range")
    if len(set(result)) != len(result):
        raise RuntimeError("duplicate denominators")

    # Verify sum is exactly 1 using fractions
    from fractions import Fraction
    total = sum(Fraction(1, d) for d in result)
    if total != 1:
        raise RuntimeError("sum is not 1")

    return result
# EVOLVE-BLOCK-END
