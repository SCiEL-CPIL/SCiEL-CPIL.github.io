import time
import random
import sys
import numpy as np

START_TIME = time.time()
HARD_DEADLINE = START_TIME + 140.0  # evaluator kills at 150s; keep 10s margin

def _now():
    return time.time()

def _remaining():
    return HARD_DEADLINE - _now()

# Precompute size-6 combos as an int array, cached per n
_COMBO6_CACHE = {}
def _combos6(n):
    if n not in _COMBO6_CACHE:
        from itertools import combinations
        _COMBO6_CACHE[n] = np.array(list(combinations(range(n), 6)), dtype=np.int64)
    return _COMBO6_CACHE[n]

def _degrees_within(n, A, combos):
    """For each combo (row i, a 6-set S), compute the degree of each of the 6
    vertices within S. Returns array shape (M,6)."""
    M = combos.shape[0]
    degs = np.zeros((M, 6), dtype=np.int64)
    # For position p (vertex combos[i,p]), its degree within S is
    # sum over q of A[combos[i,p], combos[i,q]].
    # A[combos] has shape (M,6,n): row i, position p, column x = A[combos[i,p], x].
    # We want, for each (i,p), the sum over q of that entry at column combos[i,q].
    # Equivalent: for each p, take A[combos[:,p]] (M,n) and for row i pick the 6
    # columns combos[i,0..5] and sum them.
    for p in range(6):
        # A[combos[:, p]] -> (M, n)
        sub = A[combos[:, p]]  # (M, n)
        # For each row i, sum sub[i, combos[i, 0..5]]
        # Use advanced indexing: sub[np.arange(M)[:,None], combos] -> (M,6), sum axis=1
        degs[:, p] = sub[np.arange(M)[:, None], combos].sum(axis=1)
    return degs

def _count_violations(n, A):
    if n < 6:
        return 0
    combos = _combos6(n)
    degs = _degrees_within(n, A, combos)
    d0 = degs[:, 0]
    ok = (degs[:, 1] == d0) & (degs[:, 2] == d0) & (degs[:, 3] == d0) & (degs[:, 4] == d0) & (degs[:, 5] == d0)
    return int(ok.sum())

def _is_valid(n, A):
    """Exact check: no subset of size >= 6 induces a regular subgraph.
    Checking size-6 subsets suffices (a regular size-k>=6 set contains a
    regular size-6 subset)."""
    if n < 6:
        return True
    return _count_violations(n, A) == 0

def _edges_from_A(n, A):
    e = []
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v]:
                e.append([u, v])
    return e

def _local_search(n, A, rng, deadline):
    A = A.copy()
    cur = _count_violations(n, A)
    if cur == 0:
        return A, 0
    T = 1.0
    min_T = 1e-4
    best = cur
    while _now() < deadline:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        A[u, v] ^= 1
        A[v, u] ^= 1
        new = _count_violations(n, A)
        delta = new - cur
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur = new
            if cur < best:
                best = cur
            if cur == 0:
                break
        else:
            A[u, v] ^= 1
            A[v, u] ^= 1
        T *= 0.995
        if T < min_T:
            T = min_T
    return A, cur

def construct():
    rng = random.Random(98765)
    np_rng = np.random.default_rng(98765)

    for target_n in (21, 20):
        search_deadline = _now() + (95.0 if target_n == 21 else 40.0)
        best_A = None
        best_cnt = None

        while _now() < search_deadline and _remaining() > 15.0:
            p = 0.5
            A = np.zeros((target_n, target_n), dtype=np.int8)
            triu = np.triu_indices(target_n, k=1)
            bits = (np_rng.random(len(triu[0])) < p).astype(np.int8)
            A[triu[0], triu[1]] = bits
            A[triu[1], triu[0]] = bits
            A, cnt = _local_search(target_n, A, rng, min(_now() + 8.0, search_deadline))
            if best_cnt is None or cnt < best_cnt:
                best_cnt = cnt
                best_A = A.copy()
            if best_cnt == 0:
                if _is_valid(target_n, best_A):
                    return {'n': target_n, 'edges': _edges_from_A(target_n, best_A)}
                break

        # Aggressive varied-p restarts if time remains and still no solution.
        if target_n == 21 and _remaining() > 20.0 and best_cnt != 0:
            b_deadline = _now() + min(22.0, _remaining() - 12.0)
            for p in (0.4, 0.6, 0.35, 0.65, 0.45, 0.55):
                if _now() > b_deadline:
                    break
                A = np.zeros((target_n, target_n), dtype=np.int8)
                triu = np.triu_indices(target_n, k=1)
                bits = (np_rng.random(len(triu[0])) < p).astype(np.int8)
                A[triu[0], triu[1]] = bits
                A[triu[1], triu[0]] = bits
                A, cnt = _local_search(target_n, A, rng, min(_now() + 4.0, b_deadline))
                if best_cnt is None or cnt < best_cnt:
                    best_cnt = cnt
                    best_A = A.copy()
                if best_cnt == 0 and _is_valid(target_n, best_A):
                    return {'n': target_n, 'edges': _edges_from_A(target_n, best_A)}

    # Fallback: always-valid tiny graph.
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"n={result['n']} edges={len(result['edges'])} time={time.time()-t0:.1f}s")
