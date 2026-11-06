import time
from math import factorial
from functools import lru_cache

def construct():
    start = time.time()
    deadline = start + 55  # safety margin under 60s

    N = factorial(12)  # 479001600

    # We want 1..12 distinct integers > 12 with product exactly N.
    # Minimize the largest factor.

    # Strategy: binary search on M (the max factor bound), and for each M,
    # check if we can factor N into k distinct integers all in (12, M].
    # We try from M=13 upward.

    # Key observation: the prime factors of N = 12! are limited.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Each factor must be a divisor of N, > 12, <= M, and distinct.

    # Precompute all divisors of N that are > 12
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    all_divs = get_divisors(N)
    candidates = [d for d in all_divs if d > 12]
    candidates.sort()  # ascending

    # For a given max factor M, filter candidates <= M
    # Then do DFS to pick a subset with product N, at most 12 elements, all distinct (guaranteed since candidates are unique).

    def can_factor(rem, min_idx, count, used):
        """
        rem: remaining product to achieve
        min_idx: index into candidates from which we can start
        count: how many factors chosen so far
        used: frozenset of chosen values (for distinctness - actually guaranteed by index)
        Returns list of chosen factors or None.
        """
        if time.time() > deadline:
            return None

        if rem == 1:
            return []

        if count >= 12:
            return None

        # Lower bound: the smallest possible factor we can pick is candidates[min_idx]
        # If rem < candidates[min_idx], we can't achieve rem (all remaining candidates are >= candidates[min_idx] > rem)
        # Actually we need rem to be achievable by a product of some subset of candidates[min_idx:]
        # Quick prune: if rem is not divisible by any candidate from min_idx onward, fail
        # But that's expensive. Instead, try candidates.

        # Another prune: the product of the largest (12 - count) candidates from min_idx onward must be >= rem
        # Actually, we need at least one subset whose product equals rem.
        # Let's just iterate.

        # Try each candidate from min_idx onward
        for i in range(min_idx, len(candidates)):
            c = candidates[i]
            if c > rem:
                break  # since candidates are sorted ascending
            if rem % c != 0:
                continue
            # Try including c
            result = can_factor(rem // c, i + 1, count + 1, used | {c})
            if result is not None:
                return [c] + result
        return None

    # Binary search on M: find the smallest M such that a valid factorization exists with all factors <= M.
    # Actually, since we want to minimize the largest factor, we can iterate M from 13 upward.
    # But that's slow. Better: binary search.

    # First, find an upper bound. The trivial solution is [N] but N > 12 and it's one factor.
    # Actually [N] is valid: 1 distinct integer > 12, product = N. Largest = N.
    # We want to do better.

    # Let's try a few small M values first to see if there's a quick win.
    # For M, filter candidates <= M.

    def try_with_max(M):
        """Try to find factorization with all factors <= M."""
        if time.time() > deadline:
            return None
        # Filter candidates <= M
        idx = 0
        while idx < len(candidates) and candidates[idx] <= M:
            idx += 1
        local_candidates = candidates[:idx]
        if not local_candidates:
            return None

        # Quick check: product of all local_candidates must be >= N (we need to be able to reach N)
        prod = 1
        for c in local_candidates:
            prod *= c
            if prod >= N:
                break
        if prod < N:
            return None

        # DFS
        def dfs(rem, min_idx, count):
            if time.time() > deadline:
                return None
            if rem == 1:
                return []
            if count >= 12:
                return None
            # Prune: if rem < local_candidates[min_idx] (if min_idx < len), fail
            if min_idx >= len(local_candidates):
                return None
            # Try to use a factor
            for i in range(min_idx, len(local_candidates)):
                c = local_candidates[i]
                if c > rem:
                    break
                if rem % c != 0:
                    continue
                result = dfs(rem // c, i + 1, count + 1)
                if result is not None:
                    return [c] + result
            return None

        return dfs(N, 0, 0)

    # Iterate M from 13 upward, but jump intelligently.
    # Start with a reasonable lower bound.
    # (12!)^(1/12) ≈ 38. So M must be at least ~38 if we use 12 factors.
    # But with fewer factors, M could be larger.
    # Let's try M from 13 to, say, 2000, but with jumps.

    best = None
    # Try small M first
    for M in range(13, 100):
        if time.time() > deadline:
            break
        res = try_with_max(M)
        if res is not None:
            best = res
            break  # found smallest M

    if best is None:
        # Binary search between 100 and N
        lo, hi = 100, N
        best = None
        while lo <= hi and time.time() < deadline:
            mid = (lo + hi) // 2
            res = try_with_max(mid)
            if res is not None:
                best = res
                hi = mid - 1
            else:
                lo = mid + 1

    if best is None:
        # Fallback: trivial solution
        return [N]

    # Verify
    if len(best) > 12 or any(f <= 12 for f in best) or len(set(best)) != len(best):
        return [N]
    prod = 1
    for f in best:
        prod *= f
    if prod != N:
        return [N]

    return best
