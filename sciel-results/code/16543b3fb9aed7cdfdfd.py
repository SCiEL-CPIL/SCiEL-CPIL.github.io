import time
from math import factorial

_TARGET = factorial(12)  # 479001600
_MIN = 13  # factors must be > 12
_MAX_FACTORS = 12

def _divisors(n):
    # return all divisors of n
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

def _feasible(M, deadline):
    # Can _TARGET be product of distinct ints in [_MIN, M]?
    # DFS choosing increasing factors.
    target = _TARGET
    all_divs = [d for d in _divisors(target) if _MIN <= d <= M]
    # sort
    all_divs.sort()
    
    best = [False]
    
    def dfs(remaining, start_idx, used_count, max_used):
        if time.time() > deadline:
            return
        if remaining == 1:
            best[0] = True
            return
        if used_count >= _MAX_FACTORS:
            return
        # We need at least one more factor; the smallest factor we can pick is all_divs[start_idx]
        # Prune: remaining must be >= smallest possible product of (needed) factors.
        # Simple prune: remaining must be divisible by some candidate >= start_idx and > max_used (distinct & increasing)
        for i in range(start_idx, len(all_divs)):
            d = all_divs[i]
            if d > M:
                break
            if remaining % d != 0:
                continue
            # distinct & increasing: since we go in increasing index and all_divs distinct, d > max_used automatically if i>start_idx ensures > previous? Not exactly; we need d > last chosen.
            # We'll track last chosen via start_idx ensuring increasing.
            if d <= max_used:
                continue
            if d == 1:
                continue
            dfs(remaining // d, i + 1, used_count + 1, d)
            if best[0]:
                return
    dfs(target, 0, 0, _MIN - 1)
    return best[0]

def _find_decomp(M, deadline):
    target = _TARGET
    all_divs = [d for d in _divisors(target) if _MIN <= d <= M]
    all_divs.sort()
    found = []
    
    def dfs(remaining, start_idx, used_count, chosen):
        if time.time() > deadline:
            return
        if remaining == 1:
            found.extend(chosen)
            return
        if used_count >= _MAX_FACTORS:
            return
        for i in range(start_idx, len(all_divs)):
            d = all_divs[i]
            if d > M:
                break
            if remaining % d != 0:
                continue
            if d <= chosen[-1] if chosen else False:
                continue
            if d < _MIN:
                continue
            chosen.append(d)
            dfs(remaining // d, i + 1, used_count + 1, chosen)
            chosen.pop()
            if found:
                return
    dfs(target, 0, 0, [])
    return found

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s
    
    # Lower bound for M: at least _MIN. Upper bound: _TARGET (single factor).
    # Binary search on M for feasibility.
    lo = _MIN
    hi = _TARGET
    # First ensure hi feasible (single factor _TARGET, but _TARGET > 12 and distinct count 1) -> feasible
    # Binary search smallest feasible M
    best_M = None
    # We want smallest M such that feasible.
    # But feasibility is monotonic: if feasible at M, feasible at larger M.
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if _feasible(mid, deadline):
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_M is None:
        # fallback: single factor
        return [_TARGET]
    
    # Find actual decomposition with max <= best_M
    decomp = _find_decomp(best_M, deadline)
    if not decomp:
        return [_TARGET]
    
    # Verify
    prod = 1
    for x in decomp:
        prod *= x
    assert prod == _TARGET
    assert len(decomp) == len(set(decomp))
    assert all(x > 12 for x in decomp)
    assert 1 <= len(decomp) <= 12
    return decomp
