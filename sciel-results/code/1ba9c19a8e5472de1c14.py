"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import factorial
    from functools import lru_cache
    import bisect

    deadline = time.time() + 55.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Get all divisors of target
    all_divs = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
        i += 1

    # Only divisors > 12, sorted
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)

    if n_divs == 0:
        return [target]

    def check_solution(max_val):
        """
        Checks if target can be decomposed into distinct integers from valid_divs,
        all <= max_val, with product exactly target.
        Returns list of factors if found, else None.
        """
        idx = bisect.bisect_right(valid_divs, max_val)
        candidates = valid_divs[:idx]
        n = len(candidates)

        if n == 0:
            return None

        # Precompute suffix max product bounds for pruning
        # suffix_prod[i] = product of all elements from i to n-1
        # But that's too big. Instead, we just use the fact that the product of remaining
        # must be achievable. A simple bound: if rem is not divisible by any candidate, fail.
        # Better: use memoization on (i, rem).

        @lru_cache(maxsize=None)
        def dfs(i, rem):
            if rem == 1:
                return True
            if i >= n:
                return False

            # Pruning: if rem is not divisible by candidates[i], we must skip it.
            # But we also need to check if it's possible to form rem from candidates[i:].
            # Simple bound: the smallest possible product of k distinct factors from candidates[i:]
            # is at least candidates[i] * (candidates[i]+1) * ... but that's hard.
            # Instead, rely on the divisibility check and memoization.

            # If rem is prime and greater than candidates[n-1], we can't form it.
            # But rem is always a divisor of target, so it's composite or small.

            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem):
                return True

            # Option 2: Take candidates[i]
            c = candidates[i]
            if rem % c == 0:
                if dfs(i + 1, rem // c):
                    return True

            return False

        if dfs(0, target):
            # Reconstruct solution
            result = []
            i, rem = 0, target
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                # Check if we can solve by skipping i
                if dfs(i + 1, rem):
                    # Skipped
                    i += 1
                elif rem % c == 0 and dfs(i + 1, rem // c):
                    # Took
                    result.append(c)
                    rem //= c
                    i += 1
                else:
                    # Should not happen if dfs returned True
                    break

            return result
        else:
            return None

    # Binary search for the minimal M (value in valid_divs)
    low = 0
    high = n_divs - 1
    best_result = None
    best_max = float('inf')

    while low <= high:
        if time.time() > deadline:
            break

        mid = (low + high) // 2
        max_val = valid_divs[mid]
        sol = check_solution(max_val)

        if sol is not None:
            best_result = sol
            best_max = max_val
            high = mid - 1
        else:
            low = mid + 1

    if best_result is not None:
        return best_result

    # If no solution found in time or via search, return the trivial solution
    # The trivial solution is [12!]
    return [target]

# EVOLVE-BLOCK-END
