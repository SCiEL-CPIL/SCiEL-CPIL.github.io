import time
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin under the 90s wall-clock kill

    TARGET = 479001600  # 12!

    # Valid factors: numbers of the form 2^a 3^b 5^c 7^d that are > 12.
    # (Every factor of 12! must be 7-smooth; primes > 7 cannot appear.)
    fs = set()
    for a in range(11):
        v2 = 2 ** a
        for b in range(6):
            v3 = v2 * (3 ** b)
            for c in range(3):
                v5 = v3 * (5 ** c)
                for d in range(2):
                    v = v5 * (7 ** d)
                    if v > 12:
                        fs.add(v)
    factors = sorted(fs)  # ascending, so we can break early in the DFS

    def valid(lst):
        if not (1 <= len(lst) <= 12):
            return False
        if len(set(lst)) != len(lst):
            return False
        if any(x <= 12 for x in lst):
            return False
        p = 1
        for x in lst:
            p *= x
        return p == TARGET

    def search(M, time_budget=None):
        """Return a strictly-increasing list of distinct 7-smooth factors,
        each in (12, M], whose product is TARGET; or None if impossible."""
        if time_budget is None:
            time_budget = DEADLINE - time.time()
        memo = {}  # (remaining_product, min_next) -> True once proven infeasible
        deadline = start + time_budget

        def dfs(R, lo, acc, depth):
            if time.time() > deadline:
                return None
            if R == 1:
                return acc[:] if acc else None
            key = (R, lo)
            if key in memo:
                return None
            # Try smaller factors first to find solutions quickly
            for d in factors:
                if d < lo:
                    continue
                if d > M or d > R:
                    break  # factors ascending: nothing further can work
                if R % d != 0:
                    continue
                res = dfs(R // d, d + 1, acc + [d], depth + 1)
                if res is not None:
                    return res
            memo[key] = True
            return None

        return dfs(TARGET, 13, [], 0)

    # Binary search on the max bound M: feasibility is monotone in M,
    # so we can binary search for the minimal feasible M.
    # First, find an upper bound that is feasible.
    lo, hi = 13, TARGET
    best = None
    
    # Quick check: is M=13 feasible? (Only factor 13 itself, but 13 doesn't divide 12!)
    # So start higher. Let's try a reasonable starting point.
    # We know [TARGET] works with M=TARGET, so hi is feasible.
    
    # Binary search for minimal feasible M
    while lo <= hi and time.time() < DEADLINE - 5.0:
        mid = (lo + hi) // 2
        res = search(mid, time_budget=min(10.0, DEADLINE - time.time()))
        if res is not None and valid(res):
            best = res
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best is None:
        # Fallback: try a few more values near where we left off
        for M in range(max(13, lo - 10), min(TARGET, lo + 10) + 1):
            if time.time() > DEADLINE - 2.0:
                break
            res = search(M, time_budget=2.0)
            if res is not None and valid(res):
                best = res
                break
    
    if best is None:
        # Final fallback: a single valid factor (same value as the seed program).
        return [TARGET]
    return best
