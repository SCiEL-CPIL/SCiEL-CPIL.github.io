"""Combined heuristic: multi-start local search for a high-quality reference tour,
with edge attractiveness derived from a smooth reduced-cost landscape (Hungarian
dual potentials) plus strong multiplicative promotion of reference-tour edges.
The reduced-cost landscape gives a globally coherent gradient, while the reference-tour
boost anchors the colony to a near-optimal skeleton. Uses extensive multi-start
construction with alternating 2-opt/OR-opt for high-quality reference tours."""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=500):
        """Vectorized 2-opt: compute all deltas per i at once."""
        t = np.asarray(t, dtype=int)
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

    def or_opt(t, max_passes=100):
        """Or-opt: relocate single node to best position."""
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

    # Build a strong reference tour via multi-start NN + alternating 2-opt/OR-opt
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    def refine(tour):
        nonlocal best_tour, best_len
        # Alternate 2-opt and OR-opt for better convergence
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        t = or_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    # Start from node 0
    refine(nearest_neighbor(0))

    # Randomized starts - use more starts to find better reference tour
    num_starts = 2000
    for _ in range(num_starts):
        if time.time() > deadline - 40.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    num_perturbations = 500
    for _ in range(num_perturbations):
        if time.time() > deadline - 15.0:
            break
        t = list(best_tour)
        # Perform 2-3 random 2-opt moves to perturb
        for _ in range(3):
            i = int(rng.integers(n - 1))
            j = int(rng.integers(n - 1))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0))

    # Compute Hungarian dual potentials for a smooth reduced-cost landscape
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

    # Combine: base from assignment landscape, boosted on reference tour edges
    diag = np.eye(n, dtype=bool)
    base = assignment_heur.copy()
    
    # Normalize assignment heuristic to be in a reasonable range
    base *= 100.0
    
    # Strongly promote reference tour edges using multiplicative boost on both directions
    # This preserves the relative structure of the base heuristic while strongly
    # anchoring the colony to the reference tour skeleton
    boost_factor = 500.0
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] *= boost_factor
        base[j, i] *= boost_factor
    
    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
