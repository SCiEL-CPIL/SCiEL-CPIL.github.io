"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin before 60s limit

    def search(k):
        if k < 4 or k > 256:
            return None

        def dfs(depth, min_denom, remaining, chosen):
            if time.time() > deadline:
                return None
            if depth == k:
                return chosen if remaining == 0 else None
            if remaining <= 0:
                return None

            # Max sum of (k-depth) terms each <= 1/min_denom
            if Fraction(k - depth, min_denom) < remaining:
                return None

            # a must satisfy: 1/a <= remaining => a >= ceil(1/remaining)
            # a >= min_denom
            lower = max(min_denom, (1 // remaining).numerator // remaining.denominator + (1 if (1 // remaining).numerator % remaining.denominator else 0))
            # Simpler: a >= ceil(1/remaining)
            # 1/remaining is a Fraction, ceil of it:
            # if remaining is Fraction(p, q), 1/remaining = q/p, ceil = (q + p - 1) // p
            if remaining.denominator == 0:
                return None
            lower = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)

            # Upper bound on a: we need remaining - 1/a > 0, so 1/a < remaining => a > 1/remaining
            # Also (k-depth-1)/(a+1) >= remaining - 1/a for feasibility
            # Let's just bound a: a < 1/remaining (strictly), so a <= floor(1/remaining - epsilon)
            # a_max such that 1/a_max < remaining => a_max > 1/remaining
            # So a can be at most ... well, a must be > 1/remaining, so a >= floor(1/remaining) + 1
            # Wait: 1/a < remaining => a > 1/remaining. So a >= floor(1/remaining) + 1 if 1/remaining is not integer, else 1/remaining + 1.
            # Actually a > 1/remaining, so a >= ceil(1/remaining + epsilon) = floor(1/remaining) + 1 always works as a lower bound for "a > 1/remaining"
            # Hmm, let me just set max_a based on the feasibility constraint.

            # For now, use a reasonable cap
            max_a = lower + 10000
            if lower > max_a:
                return None

            for a in range(lower, max_a + 1):
                if time.time() > deadline:
                    return None
                one_over_a = Fraction(1, a)
                new_remaining = remaining - one_over_a
                if new_remaining < 0:
                    break
                if new_remaining == 0:
                    if depth + 1 == k:
                        return chosen + [a]
                    continue
                # Check feasibility: (k-depth-1) terms each <= 1/(a+1)
                if Fraction(k - depth - 1, a + 1) < new_remaining:
                    continue
                result = dfs(depth + 1, a + 1, new_remaining, chosen + [a])
                if result is not None:
                    return result
            return None

        return dfs(0, 5, Fraction(1, 1), [])

    for k in range(4, 20):
        result = search(k)
        if result is not None:
            return result
        if time.time() > deadline:
            break

    raise RuntimeError("no representation found within deadline")
# EVOLVE-BLOCK-END
