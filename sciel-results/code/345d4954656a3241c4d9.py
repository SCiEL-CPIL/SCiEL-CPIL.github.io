import time
import math
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85.0

    def try_k(k):
        """Search for k distinct denominators >= 5 summing to 1."""
        if time.time() > deadline:
            return None

        target = Fraction(1, 1)

        def dfs(remaining, min_den, depth, chosen):
            if time.time() > deadline:
                return None

            if depth == k:
                return list(chosen) if remaining == 0 else None

            rem_terms = k - depth

            if remaining <= 0:
                return None

            # Lower bound: d >= ceil(1/remaining)
            lo = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)

            # Upper bound: d <= floor(rem_terms / remaining)
            hi = (rem_terms * remaining.denominator) // remaining.numerator
            hi = min(hi, 10_000_000)

            if lo > hi:
                return None

            # Pruning: max sum of rem_terms unit fractions starting from lo
            # Must be >= remaining
            max_sum = Fraction(0, 1)
            for i in range(rem_terms):
                max_sum += Fraction(1, lo + i)
            if remaining > max_sum:
                return None

            for d in range(lo, hi + 1):
                new_rem = remaining - Fraction(1, d)
                if new_rem < 0:
                    break
                chosen.append(d)
                res = dfs(new_rem, d + 1, depth + 1, chosen)
                if res is not None:
                    return res
                chosen.pop()

            return None

        return dfs(target, 5, 0, [])

    # Try k=8 first (minimum possible), then increase
    for k in [8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline:
            break
        result = try_k(k)
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(result) == len(set(result)) and all(5 <= d <= 10_000_000 for d in result):
                return result

    # Fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
