"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache
from itertools import product

TARGET = factorial(12)  # 479001600

# Prime factorization: 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
PRIMES = [2, 3, 5, 7, 11]
EXPS = [10, 5, 2, 1, 1]


def _build_divisors(max_val):
    """Generate all divisors of TARGET that are > 12 and <= max_val, sorted ascending."""
    divisors = []
    n = len(PRIMES)

    def gen(idx, cur):
        if cur > max_val:
            return
        if idx == n:
            if cur > 12:
                divisors.append(cur)
            return
        p = PRIMES[idx]
        for e in range(EXPS[idx] + 1):
            gen(idx + 1, cur * (p ** e))

    gen(0, 1)
    divisors.sort()
    return divisors


def _search_partition(divs, target, max_factors, deadline):
    """
    Find a subset of distinct divisors from divs (each <= some bound) whose product is target,
    using at most max_factors elements. Returns the list or None.
    Uses DFS with memoization and time checking.
    """
    n = len(divs)
    memo = {}
    found = [None]

    def dfs(idx, tgt, rem):
        if time.time() > deadline:
            return None
        if tgt == 1:
            return []
        if rem == 0 or idx >= n:
            return None
        key = (idx, tgt, rem)
        if key in memo:
            return memo[key]
        for i in range(idx, n):
            d = divs[i]
            if d > tgt:
                break
            if tgt % d != 0:
                continue
            sub = dfs(i + 1, tgt // d, rem - 1)
            if sub is not None:
                memo[key] = [d] + sub
                return [d] + sub
        memo[key] = None
        return None

    result = dfs(0, target, max_factors)
    return result


def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill

    # Strategy: sweep candidate max-factor bounds in increasing order.
    # For each bound B (a divisor of TARGET > 12), check if there exists a subset of
    # distinct divisors (each in (12, B]) whose product is TARGET.
    # The first such B that works gives the minimal possible largest factor.

    # Get all divisors of TARGET > 12, sorted.
    all_divs = _build_divisors(TARGET)
    n_divs = len(all_divs)

    # We'll iterate over candidate bounds. Since all_divs is sorted, for bound = all_divs[k],
    # the valid divisors are all_divs[0:k+1].
    # To avoid trying every single divisor (could be thousands), we'll try a subset
    # but start from the smallest. Actually, let's try all of them but with a time check.
    # The number of divisors of 12! is (10+1)(5+1)(2+1)(1+1)(1+1) = 11*6*3*2*2 = 792.
    # Minus those <= 12: 1,2,3,4,5,6,8,9,10,12 -> 10 divisors. So ~782 valid divisors.
    # Trying 782 bounds, each with a DFS... the DFS might be expensive for larger bounds.
    # But for small bounds, the divisor list is small, so DFS is fast.
    # Let's try.

    for k in range(n_divs):
        if time.time() > deadline - 2.0:
            break

        bound = all_divs[k]
        divs_subset = all_divs[:k + 1]

        # Quick prune: if bound is too small to possibly work?
        # With at most 12 factors each <= bound, product <= bound^12.
        # If bound^12 < TARGET, skip. But bound is at least 13, 13^12 is huge.
        # Actually, the issue is more about whether the specific divisors can multiply to TARGET.
        # Let's just run the DFS.

        # Try to find a partition. We'll allow up to 12 factors.
        result = _search_partition(divs_subset, TARGET, 12, deadline - 1.0)

        if result is not None:
            # Found a valid partition with all factors <= bound.
            # Since we're iterating bounds in increasing order, this is the minimal bound.
            # The actual largest factor in result is max(result), which is <= bound.
            # But could there be a solution with max factor < bound?
            # No, because if there were, it would have been found at an earlier bound.
            # So max(result) is the minimal possible largest factor.
            return result

    # Fallback: return [TARGET] if no better solution found (shouldn't happen)
    return [TARGET]
# EVOLVE-BLOCK-END
