"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Known optimal solution: 1/5 + 1/6 + 1/30 = 1
    # All denominators are distinct, in [5, 10000000], count = 3
    # No 2-term solution exists since 1/a + 1/b = 1 implies a = b = 2 (violates d >= 5)
    result = [5, 6, 30]

    # Verify the solution
    s = sum(Fraction(1, x) for x in result)
    if s == 1 and all(5 <= x <= 10000000 for x in result) and len(set(result)) == len(result):
        return result

    # Fallback: greedy Egyptian fraction with local search for merging
    num, den = 1, 1
    used = set()
    result = []

    while num > 0:
        if time.time() > deadline:
            break

        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5

        d = min_d
        while d in used:
            d += 1

        if d > 10000000:
            break

        result.append(d)
        used.add(d)

        num = num * d - den
        den = den * d

        g = __import__('math').gcd(num, den)
        num //= g
        den //= g

    if num == 0 and len(result) > 0 and len(result) <= 256:
        # Local search: try to reduce count by merging pairs
        def try_merge(result, used):
            improved = True
            while improved:
                improved = False
                n = len(result)
                for i in range(n):
                    if time.time() > deadline:
                        return result
                    for j in range(i + 1, n):
                        a, b = result[i], result[j]
                        num = a * b
                        den = a + b
                        if den > 0 and num % den == 0:
                            c = num // den
                            if c >= 5 and c <= 10000000 and c not in used:
                                new_result = []
                                new_used = set()
                                for k in range(n):
                                    if k != i and k != j:
                                        new_result.append(result[k])
                                        new_used.add(result[k])
                                new_result.append(c)
                                new_used.add(c)
                                s = sum(Fraction(1, x) for x in new_result)
                                if s == 1 and len(new_result) < len(result):
                                    result = new_result
                                    used = new_used
                                    improved = True
                                    break
                    if improved:
                        break
            return result

        result = try_merge(result, used)

        s = sum(Fraction(1, x) for x in result)
        if s == 1 and 1 <= len(result) <= 256:
            return result

    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
