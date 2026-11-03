"""QAP solver for QAPLIB tho150 via iterated 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost(a, b, p):
    return int(np.einsum("ij,pi,pj->", a, b, p))


def _swap_delta(a, b, p, i, j, S_col, S_row):
    u = p[i]
    v = p[j]
    if u == v:
        return 0
    Au = a[u]
    Av = a[v]
    Bu = b[u]
    Bv = b[v]
    return (S_col[i] - S_col[j]) * (Au - Av) + (S_row[i] - S_row[j]) * (Bu - Bv) \
        + 2 * (Au @ Bv + Av @ Bu - Au @ Bu - Av @ Bv)


def _local_search(a, b, p, S_col, S_row, cost, deadline, max_passes=200):
    n = len(p)
    # Precompute per-facility row/col sums of a (constant across the search).
    a_row = a.sum(axis=1)
    a_col = a.sum(axis=0)
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        # First-improvement 2-opt with a light random restart of scan order per pass
        order = np.random.permutation(n)
        for i in order:
            if time.time() > deadline:
                return p, cost
            for j in order:
                if j <= i:
                    continue
                d = _swap_delta(a, b, p, i, j, S_col, S_row)
                if d < 0:
                    # Apply swap
                    p[i], p[j] = p[j], p[i]
                    # Update S_col / S_row in O(n)
                    u = p[i]
                    v = p[j]
                    # S_col[k] = sum_m a[m, p[k]]  -> depends on p[k]
                    S_col[i] = a_col[u]
                    S_col[j] = a_col[v]
                    # S_row[k] = sum_m a[p[k], m] -> depends on p[k]
                    S_row[i] = a_row[u]
                    S_row[j] = a_row[v]
                    cost += d
                    improved = True
    return p, cost


def _perturb(p, k):
    n = len(p)
    if k <= 0:
        return
    idx = np.random.choice(n, size=k, replace=False)
    # Randomly swap pairs
    for t in range(0, len(idx) - 1, 2):
        i, j = idx[t], idx[t + 1]
        p[i], p[j] = p[j], p[i]


def solve(a, b):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150 s kill
    n = len(a)

    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    S_col = np.array([a[:, p[k]] @ np.ones(n) for k in range(n)], dtype=np.int64)
    # Simpler: S_col[k] = a[:, p[k]].sum()
    S_col = np.array([a[:, p[k]].sum() for k in range(n)], dtype=np.int64)
    S_row = np.array([a[p[k], :].sum() for k in range(n)], dtype=np.int64)

    cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Phase 1: polish the initial solution
    p, cost = _local_search(a, b, p, S_col, S_row, cost, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()

    # Phase 2: iterated local search
    # Estimate per-iteration cost; budget remaining time.
    while time.time() < deadline - 5.0:
        # Perturb a copy of the current best
        cur = best_p.copy()
        S_col_c = np.array([a[:, cur[k]].sum() for k in range(n)], dtype=np.int64)
        S_row_c = np.array([a[cur[k], :].sum() for k in range(n)], dtype=np.int64)
        cur_cost = best_cost
        # Perturb amount grows mildly over time to escape deeper
        k = max(2, n // 30)
        _perturb(cur, k)
        cur_cost = _cost(a, b, cur)
        S_col_c = np.array([a[:, cur[k]].sum() for k in range(n)], dtype=np.int64)
        S_row_c = np.array([a[cur[k], :].sum() for k in range(n)], dtype=np.int64)

        cur, cur_cost = _local_search(a, b, cur, S_col_c, S_row_c, cur_cost, deadline)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = cur.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
