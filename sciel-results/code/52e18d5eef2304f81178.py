import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit

    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    target = 479001600
    target_exps = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]

    # Generate all divisors of 12! that are > 12
    divisors = []
    for e1 in range(11):
        for e2 in range(6):
            for e3 in range(3):
                for e4 in range(2):
                    for e5 in range(2):
                        d = (2 ** e1) * (3 ** e2) * (5 ** e3) * (7 ** e4) * (11 ** e5)
                        if d > 12:
                            divisors.append((d, (e1, e2, e3, e4, e5)))

    divisors.sort()
    n_divs = len(divisors)

    # Fallback
    if time.time() > deadline:
        return [target]

    # Binary search on the maximum factor M
    # For a given max_index (divisors up to this index are allowed),
    # check if we can form target using a subset of those divisors.

    def can_form(max_idx):
        """Check if target can be formed using a subset of divisors[0:max_idx+1]."""
        # DP: set of reachable exponent vectors
        reachable = {target_exps}
        for i in range(max_idx + 1):
            d_val, d_exps = divisors[i]
            new_states = set()
            for rem in reachable:
                # Check if rem >= d_exps component-wise
                if all(rem[j] >= d_exps[j] for j in range(5)):
                    nr = tuple(rem[j] - d_exps[j] for j in range(5))
                    new_states.add(nr)
            if (0, 0, 0, 0, 0) in new_states:
                return True
            reachable.update(new_states)
        return (0, 0, 0, 0, 0) in reachable

    # Binary search for the smallest max_idx such that can_form(max_idx) is True
    lo, hi = 0, n_divs - 1
    best_idx = n_divs - 1  # fallback: use all divisors

    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if can_form(mid):
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if time.time() > deadline:
        return [target]

    # Reconstruct: find which divisors are used
    # Use parent tracking with 0/1 knapsack DP
    max_idx = best_idx
    start_rem = target_exps
    parent_map = {start_rem: (None, None)}  # rem -> (prev_rem, div_index)
    reachable = {start_rem}

    for i in range(max_idx + 1):
        if time.time() > deadline:
            break
        d_val, d_exps = divisors[i]
        # Snapshot of current reachable states
        snapshot = list(reachable)
        for rem in snapshot:
            if all(rem[j] >= d_exps[j] for j in range(5)):
                nr = tuple(rem[j] - d_exps[j] for j in range(5))
                if nr not in reachable:
                    reachable.add(nr)
                    parent_map[nr] = (rem, i)
        if (0, 0, 0, 0, 0) in reachable:
            break

    if (0, 0, 0, 0, 0) not in reachable:
        return [target]

    # Reconstruct factors
    factors = []
    curr = (0, 0, 0, 0, 0)
    while parent_map[curr][1] is not None:
        prev_rem, div_idx = parent_map[curr]
        factors.append(divisors[div_idx][0])
        curr = prev_rem

    return factors
