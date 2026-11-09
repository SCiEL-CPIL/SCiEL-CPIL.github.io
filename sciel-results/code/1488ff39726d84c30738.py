"""
Heuristic: multi-start local search reference tour + Hungarian reduced-cost landscape.
Adds MST-based shortest-path guidance to the base heuristic for smooth connectivity
structure, while keeping the strong reference-tour anchoring.
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path as sp_shortest_path

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 285.0

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=300):
        t = np.asarray(t, dtype=int).copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                if i == 0:
                    j_start, j_end = 2, n - 1
                else:
                    j_start, j_end = i + 2, n
                if j_start >= j_end:
                    continue
                js = np.arange(j_start, j_end, dtype=int)
                a = t[i]
                b = t[i + 1]
                cs = t[js]
                ds = t[(js + 1) % n]
                deltas = (D[a, cs] + D[b, ds]) - (D[a, b] + D[cs, ds])
                mask = deltas < -1e-12
                if np.any(mask):
                    j = js[np.argmax(mask)]
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        return t

    def or_opt(t, max_passes=40):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for k in range(n):
                node = t[k]
                prev = t[(k - 1) % n]
                nxt = t[(k + 1) % n]
                cost_remove = D[prev, node] + D[node, nxt] - D[prev, nxt]
                best_delta = cost_remove
                best_pos = -1
                for pos in range(n - 1):
                    if pos == k or pos == (k - 1) % n:
                        continue
                    a = t[pos]
                    b = t[(pos + 1) % n]
                    if a == prev or b == prev or a == nxt or b == nxt:
                        continue
                    cost_insert = D[a, node] + D[node, b] - D[a, b]
                    delta = cost_remove + cost_insert
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_pos = pos
                if best_pos >= 0:
                    t.pop(k)
                    t.insert(best_pos + 1, node)
                    improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    def refine(tour):
        nonlocal best_tour, best_len
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    refine(nearest_neighbor(0))

    num_starts = 200
    for _ in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    for _ in range(80):
        if time.time() > deadline - 8.0:
            break
        t = list(best_tour)
        for _ in range(3):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0))

    # Hungarian reduced-cost landscape
    try:
        row_ind, col_ind = linear_sum_assignment(D)
        perm = np.zeros(n, dtype=int)
        perm_inv = np.zeros(n, dtype=int)
        for i, j in zip(row_ind, col_ind):
            perm[i] = j
            perm_inv[j] = i
        row_cost = np.array([D[i, perm[i]] for i in range(n)])
        col_cost = np.array([D[perm_inv[j], j] for j in range(n)])
        regret = D - (row_cost[:, None] + col_cost[None, :]) / 2.0
        regret = np.maximum(regret, 0.0)
        scale = 5.0
        assignment_heur = np.exp(-regret * scale)
    except Exception:
        assignment_heur = 1.0 / np.maximum(D, 1e-9)
        np.fill_diagonal(assignment_heur, 0.0)

    # MST shortest-path guidance: penalize edges that are far apart in the MST
    # This provides a smooth global structure that avoids local loops
    mst_heur = np.ones((n, n))
    try:
        # Build MST from distance matrix (treating non-diagonal as edges)
        D_sparse = D.copy()
        np.fill_diagonal(D_sparse, 0)
        mst = minimum_spanning_tree(D_sparse)
        # Compute shortest paths within the MST
        mst_dist = sp_shortest_path(mst)
        # Heuristic: exp(-mst_distance * scale_mst)
        # Smaller MST distance -> more attractive
        scale_mst = 3.0
        mst_heur = np.exp(-mst_dist * scale_mst)
        mst_heur[~np.isfinite(mst_heur)] = 1e-9
        mst_heur = np.maximum(mst_heur, 1e-9)
    except Exception:
        mst_heur = np.ones((n, n))
        np.fill_diagonal(mst_heur, 0.0)

    # Combine: weighted product of assignment heuristic and MST heuristic
    # Use geometric mean (product of square roots) for smooth combination
    diag = np.eye(n, dtype=bool)
    base = np.sqrt(assignment_heur * mst_heur)

    # Scale up
    base *= 100.0

    # Strongly promote reference tour edges
    boost_factor = 800.0
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] = max(base[i, j], 10000.0)
        base[j, i] = max(base[j, i], 10000.0)

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
