import time
from math import factorial
from functools import lru_cache

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s

    N = factorial(12)  # 479001600

    # Precompute all divisors of N
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    all_divs = get_divisors(N)
    candidates = [d for d in all_divs if d > 12]
    candidates.sort()
    candidate_set = set(candidates)

    # Precompute divisors of N for quick lookup
    n_div_set = set(all_divs)

    def try_with_max(M):
        """Try to find factorization with all factors <= M, all distinct, all > 12."""
        if time.time() > deadline:
            return None

        # Filter candidates <= M
        local_candidates = [c for c in candidates if c <= M]
        if not local_candidates:
            return None

        # Quick feasibility: product of all local candidates must be >= N
        # (we need to be able to reach N from some subset)
        # Actually this isn't quite right - we need a subset whose product is exactly N.
        # Better check: N must be achievable. Let's just try DFS.

        # Precompute, for each divisor d of N that is <= M, the list of divisors of d
        # that are also in local_candidates (and > 12)
        # This is expensive, so we'll compute on the fly but cache.

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def divisors_in_range(rem):
            """Return sorted list of divisors of rem that are in local_candidates."""
            result = []
            # rem must be a divisor of N for this to be useful
            if rem not in n_div_set:
                return []
            for d in all_divs:
                if d > rem:
                    break
                if rem % d == 0 and d > 12 and d <= M:
                    result.append(d)
            return result

        def dfs(rem, min_val, count, current):
            if time.time() > deadline:
                return None

            if rem == 1:
                if 1 <= len(current) <= 12:
                    return list(current)
                return None

            if count >= 12:
                return None

            # Get divisors of rem that are >= min_val and <= M
            # and are divisors of N
            # We need them to be in local_candidates
            for d in divisors_in_range(rem):
                if d < min_val:
                    continue
                # d is a divisor of rem, d > 12, d <= M, d is a divisor of N
                current.append(d)
                result = dfs(rem // d, d + 1, count + 1, current)
                if result is not None:
                    return result
                current.pop()

            return None

        return dfs(N, 13, 0, [])

    best = None

    # Try small M values first - solutions with small max are more constrained and faster to find
    # Lower bound on M: at minimum, we need factors > 12. If we use k factors, each > 12,
    # the product is > 12^k. For k=12: 12^12 is way too big. For k=2: need two factors > 12 with product N.
    # The smallest possible max is when we split N into factors as evenly as possible.
    # N^(1/2) ≈ 21886, N^(1/3) ≈ 783, N^(1/4) ≈ 262, N^(1/5) ≈ 137, N^(1/6) ≈ 91, N^(1/7) ≈ 67, N^(1/8) ≈ 52, N^(1/9) ≈ 43, N^(1/10) ≈ 37, N^(1/11) ≈ 33, N^(1/12) ≈ 30.

    # Try M from 13 upward, but skip to reasonable values
    # For 2 factors: M >= ceil(N/13) ≈ 36846
    # For 3 factors: M >= N^(1/3) ≈ 783
    # For 4 factors: M >= N^(1/4) ≈ 262
    # For 5 factors: M >= N^(1/5) ≈ 137
    # For 6 factors: M >= N^(1/6) ≈ 91
    # For 7 factors: M >= N^(1/7) ≈ 67
    # For 8 factors: M >= N^(1/8) ≈ 52
    # For 9 factors: M >= N^(1/9) ≈ 43
    # For 10 factors: M >= N^(1/10) ≈ 37
    # For 11 factors: M >= N^(1/11) ≈ 33
    # For 12 factors: M >= N^(1/12) ≈ 30

    # So the absolute lower bound on M is around 30 (for 12 factors).
    # Let's try M from 30 upward in steps.

    # First pass: try M from 30 to 200
    for M in range(30, 201):
        if time.time() > deadline:
            break
        res = try_with_max(M)
        if res is not None:
            best = res
            break

    if best is None:
        # Second pass: try M from 200 to 1000
        for M in range(200, 1001, 10):
            if time.time() > deadline:
                break
            res = try_with_max(M)
            if res is not None:
                best = res
                break

    if best is None:
        # Third pass: binary search between 1000 and N
        lo, hi = 1000, N
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
    if len(best) > 12 or len(best) < 1 or any(f <= 12 for f in best) or len(set(best)) != len(best):
        return [N]
    prod = 1
    for f in best:
        prod *= f
    if prod != N:
        return [N]

    return best
