import time
from itertools import combinations

# ---- core helpers ----
def _bad(a, b):
    return (a * b) % (a + b) == 0

def _is_valid(S):
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            if _bad(a, S[j]):
                return False
    return True

def _greedy_add(S, pool, deadline):
    """Greedily add elements from pool that stay compatible, maximizing size."""
    S = list(S)
    for x in pool:
        if time.time() > deadline:
            break
        if all(not _bad(x, y) for y in S):
            S.append(x)
    return S

def _local_search(S, pool, deadline, iters=200000):
    """1- and 2-exchange local search: try removing one and adding another."""
    S = list(S)
    Sset = set(S)
    improved = True
    it = 0
    while improved and it < iters and time.time() < deadline:
        improved = False
        it += 1
        # try single swap: remove r, add a
        for a in pool:
            if a in Sset:
                continue
            for r in S:
                if _bad(a, r):
                    continue
                # check a vs all others except r
                ok = True
                for y in S:
                    if y == r:
                        continue
                    if _bad(a, y):
                        ok = False
                        break
                if ok:
                    S = [y for y in S if y != r]
                    S.append(a)
                    Sset = set(S)
                    improved = True
                    break
            if improved:
                break
    return S

def _build_fallback():
    """Deterministic strong set: odds + greedy-compatible evens, then local search."""
    deadline = time.time() + 25.0
    odds = list(range(1, 257, 2))
    evens = list(range(2, 257, 2))
    # start from odds, add compatible evens greedily
    S = _greedy_add(odds, evens, deadline)
    # local search with remaining evens as pool
    pool = [e for e in evens if e not in S]
    S = _local_search(S, pool, deadline, iters=100000)
    # ensure sorted
    S = sorted(S)
    if not _is_valid(S):
        # ultimate fallback: pure odds (always valid)
        S = list(range(1, 257, 2))
    return S

# ---- build once at import, with a hard cap ----
_T0 = time.time()
_FALLBACK = None
try:
    _FALLBACK = _build_fallback()
except Exception:
    _FALLBACK = list(range(1, 257, 2))

def construct():
    # hard guard: if we've already spent too long, return guaranteed-valid odds
    if time.time() - _T0 > 55.0:
        return list(range(1, 257, 2))
    S = _FALLBACK
    if S is None or not _is_valid(S) or len(S) == 0:
        return list(range(1, 257, 2))
    return list(S)
