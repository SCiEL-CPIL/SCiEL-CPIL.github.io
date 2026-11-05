import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before the 90s kill

TARGET = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11


def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)


def _all_divisors(n):
    """Return all divisors of n, sorted."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs


# Precompute all divisors of 12! that are > 12
_ALL_DIVS = [d for d in _all_divisors(TARGET) if d > 12]


def _can_solve(limit):
    """
    Determine if there exists a valid decomposition of TARGET into
    1..12 distinct factors all in (12, limit] with product TARGET.
    Returns the list of factors if found, else None.
    """
    if _time_left() < 3:
        return None

    cands = [d for d in _ALL_DIVS if d <= limit]
    if not cands:
        return None

    n = len(cands)
    path = []
    # For pruning: precompute suffix products of smallest candidates
    # suffix_prod[i] = product of cands[i], cands[i+1], ..., cands[n-1]
    # If suffix_prod[start_idx] > remaining, we can't fill with smallest
    # But that's too pessimistic. Instead, use: if remaining > suffix_prod[start_idx],
    # we need all of them. Actually, we need product of chosen subset = remaining.
    # A better prune: the maximum product we can get from cands[start_idx:] is
    # the product of all of them. If that < remaining, fail.
    # The minimum product (with at least 1 factor) is cands[start_idx].
    # If cands[start_idx] > remaining, fail (can't even pick the smallest).
    # Actually we need exactly remaining. Let's use:
    #   If remaining == 1: success (we've factored everything)
    #   If remaining < cands[start_idx]: fail (smallest candidate too big)
    #   If product of ALL cands[start_idx:] < remaining: fail (can't reach remaining)

    # Precompute suffix products
    suffix_prod = [1] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * cands[i]

    def dfs(start_idx, remaining):
        if _time_left() < 1:
            return None

        if remaining == 1:
            return list(path) if path else None

        if len(path) >= 12:
            return None

        if start_idx >= n:
            return None

        # Prune: smallest available candidate
        if cands[start_idx] > remaining:
            return None

        # Prune: even using all remaining candidates, can't reach 'remaining'
        if suffix_prod[start_idx] < remaining:
            return None

        for i in range(start_idx, n):
            d = cands[i]
            # If d > remaining, and cands is sorted, we can break
            if d > remaining:
                break
            # remaining must be divisible by d
            if remaining % d != 0:
                continue

            path.append(d)
            result = dfs(i + 1, remaining // d)
            if result is not None:
                return result
            path.pop()

        return None

    return dfs(0, TARGET)


def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor; the number of factors is variable.
    """
    if _time_left() < 5:
        return [TARGET]

    # Binary search for the minimal M such that a valid decomposition exists
    # with all factors in (12, M].
    # The property "exists solution with max <= M" is monotonic in M.
    lo = 13
    hi = TARGET  # trivial solution: [TARGET]

    best_sol = None

    while lo < hi:
        if _time_left() < 5:
            break
        mid = (lo + hi) // 2
        sol = _can_solve(mid)
        if sol is not None:
            best_sol = sol
            hi = mid
        else:
            lo = mid + 1

    # After binary search, lo == hi is the minimal M.
    # Get the actual solution for this M.
    if best_sol is not None and max(best_sol) <= lo:
        return best_sol

    final_sol = _can_solve(lo)
    if final_sol is not None:
        return final_sol

    # Fallback: return the trivial solution
    return [TARGET]
