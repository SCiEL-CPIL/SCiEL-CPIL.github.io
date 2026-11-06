import math
import time

T12 = math.factorial(12)          # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = (10, 5, 2, 1, 1)      # 2^10 3^5 5^2 7 11

# ---- exponent-vector helpers ---------------------------------------------
def _exp_vector(n):
    v = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            n //= p
            v[i] += 1
    return tuple(v)

# Precompute every divisor of T12 that is > 12, with its exponent vector.
DIVS = []          # sorted divisors > 12
DIVS_EV = []       # parallel exponent vectors
def _init_divisors():
    limit = math.isqrt(T12)
    seen = set()
    for i in range(1, limit + 1):
        if T12 % i == 0:
            for d in (i, T12 // i):
                if d > 12 and d not in seen:
                    seen.add(d)
                    DIVS.append(d)
    DIVS.sort()
    for d in DIVS:
        DIVS_EV.append(_exp_vector(d))

_init_divisors()
N_DIVS = len(DIVS)

def _valid(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f <= 12 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12

# ---- explicit incumbent (from prime-power grouping) -----------------------
# 144=2^4*3^2, 64=2^6, 77=7*11, 25=5^2, 27=3^3  -> 2^10 3^5 5^2 7 11
CANDIDATE = [144, 64, 77, 25, 27]

# ---- cheap feasibility: can M divisors (<=M) supply the full exponent set? -
def _feasible(M):
    """True if some subset of divisors <= M has exponent sum == EXP_TOTAL."""
    # gather usable divisors
    usable = [(DIVS[i], DIVS_EV[i]) for i in range(N_DIVS) if DIVS[i] <= M]
    if not usable:
        return False
    # quick upper bound: sum of available exponents per prime
    tot = [0, 0, 0, 0, 0]
    for _, ev in usable:
        for k in range(5):
            tot[k] += ev[k]
    for k in range(5):
        if tot[k] < EXP_TOTAL[k]:
            return False
    # DFS subset-sum over exponent vectors (take/skip), bounded by remaining need.
    # Order by descending "value" to find a solution fast.
    usable.sort(key=lambda x: x[1][0] * 1000 + x[1][1] * 100 + x[1][2], reverse=True)
    target = list(EXP_TOTAL)
    n = len(usable)

    def dfs(pos, rem):
        if all(r == 0 for r in rem):
            return True
        if pos >= n:
            return False
        # prune: remaining usable exponents can't cover rem
        d, ev = usable[pos]
        # take
        ok = True
        for k in range(5):
            if ev[k] > rem[k]:
                ok = False
                break
        if ok:
            nr = [rem[k] - ev[k] for k in range(5)]
            if dfs(pos + 1, nr):
                return True
        # skip
        if dfs(pos + 1, rem):
            return True
        return False

    return dfs(0, target)

# ---- focused DFS to find a set with max factor <= M -----------------------
def _search(M, deadline, best):
    """DFS over divisors in increasing order, each > previous, all <= M,
    product == T12. Returns a found list or None."""
    # candidate indices: divisors in (12, M], sorted ascending
    idx = [i for i in range(N_DIVS) if 13 <= DIVS[i] <= M]
    if not idx:
        return None
    rem = list(EXP_TOTAL)
    sol = []

    def dfs(pos, rem, last_div):
        if time.time() > deadline:
            return False
        if all(r == 0 for r in rem):
            return True
        if pos >= len(idx):
            return False
        # iterate candidates from pos, in increasing order, > last_div
        for j in range(pos, len(idx)):
            i = idx[j]
            d = DIVS[i]
            if d <= last_div:
                continue
            if d > M:
                break
            ev = DIVS_EV[i]
            ok = True
            for k in range(5):
                if ev[k] > rem[k]:
                    ok = False
                    break
            if not ok:
                continue
            sol.append(d)
            if dfs(j + 1, [rem[k] - ev[k] for k in range(5)], d):
                return True
            sol.pop()
            if time.time() > deadline:
                return False
        return False

    if dfs(0, rem, 12):
        return list(sol)
    return None

def construct():
    start = time.time()
    deadline = start + 82.0          # safety margin under 90s

    # incumbent from explicit construction
    if _valid(CANDIDATE):
        best = list(CANDIDATE)
    else:
        best = [T12]
    best_max = max(best)

    # local improvement: try to lower the max factor by replacing it
    def try_local_improve(best, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_max = max(best)
            others = [f for f in best if f != cur_max]
            rem = T12 // cur_max
            # a single replacement keeps product only if d == cur_max; instead
            # we try to find a *different* set via _search with tighter bound.
            for M in range(cur_max - 1, 12, -1):
                if time.time() > deadline:
                    break
                found = _search(M, deadline, best)
                if found and _valid(found):
                    best = found
                    best_max = max(best)
                    improved = True
                    break
        return best, best_max

    best, best_max = try_local_improve(best, deadline)

    # binary search for the minimal feasible M using the cheap feasibility test,
    # then a focused search to realize it.
    lo, hi = 13, best_max
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if _feasible(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    target_M = lo
    if target_M < best_max and time.time() < deadline:
        found = _search(target_M, deadline, best)
        if found and _valid(found):
            best = found
            best_max = max(best)

    # final safety: return a valid set
    if not _valid(best):
        return [T12]
    return best
