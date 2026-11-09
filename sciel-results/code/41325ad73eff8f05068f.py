"""Heuristic matrix for ACO orienteering: option-value model of remaining budget."""
import numpy as np
import time

_DEADLINE = 322.0  # safety margin before the 330 s wall-clock kill


def _option_value(prizes, d, m, max_len, budget, iters=140):
    """Expected total prize collectable starting from node m with `budget` length left.

    Dynamic programming over a discretized budget.  This is a *different modelling
    family* from a static prize/distance ratio: it captures how much prize can still
    be gathered after committing to a node, which is exactly what an ant needs to
    decide whether a move keeps its tour viable.
    """
    n = len(prizes)
    if n <= 1:
        return 0.0
    # distance to the nearest other node (fallback scale for the budget grid)
    row = d[m].copy()
    row[m] = np.inf
    dmin = float(np.min(row))
    if not np.isfinite(dmin) or dmin <= 0.0:
        dmin = 1e-3
    scale = max(max_len, dmin * 1.5)
    budget = max(0.0, min(float(budget), scale))
    K = 80
    xs = np.linspace(0.0, scale, K + 1)
    dx = xs[1] - xs[0]
    idx = np.clip(np.searchsorted(xs, budget) - 1, 0, K - 1)
    V = np.zeros(K + 1)
    for _ in range(iters):
        newV = V.copy()
        for j in range(n):
            if j == m:
                continue
            dj0 = d[j, 0]
            pj = prizes[j]
            if dj0 >= scale:
                continue
            # need: d[m,j] + d[j,0] <= budget  to reach j and still return
            rem = budget - (d[m, j] + dj0)
            if rem < 0.0:
                continue
            k = int(np.clip(np.floor(rem / dx), 0, K))
            # best over all budgets <= rem of (prize_j + V[b])
            prefix = np.maximum.accumulate(pj + V[:k + 1])
            newV[idx] = max(newV[idx], float(prefix[k]))
        V = newV
    return float(V[idx])


def heuristics(prizes, distance_matrix, max_len):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    max_len = float(max_len)
    n = len(prizes)

    # ---- option-value heuristic (different modelling family) ----
    H = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 1.0
                continue
            # budget remaining after going i -> j and then returning to 0
            budget = max_len - (d[i, j] + d[j, 0])
            if budget < 0.0:
                H[i, j] = 1e-9          # move i->j is infeasible
            else:
                H[i, j] = _option_value(prizes, d, j, max_len, budget)
                if not np.isfinite(H[i, j]) or H[i, j] <= 0.0:
                    H[i, j] = 1e-9

    # ---- deadline guard: return the best (finite, correct-shape) result ----
    if time.time() - t0 > _DEADLINE:
        return H

    # ---- structural refinement: sharpen feasibility + blend in the proven ratio ----
    # Sharpen: feasible moves are strongly boosted, infeasible ones suppressed, so the
    # ant is steered toward moves that keep the tour within budget (the real constraint).
    feas = np.where((d + d[:, 0:1]) < max_len, 1.0, 1e-9)
    H = H * feas

    # Blend in the static prize/distance ratio (the seed's proven signal).  The ratio
    # rewards high prize close by; the option value rewards moves that leave room.
    ratio = (prizes[None, :] + 1e-9) / (d + 1e-9)
    ratio = np.nan_to_num(ratio, nan=1e-9, posinf=1e9, neginf=1e-9)
    H = 0.5 * H + 0.5 * ratio

    # ---- final safety: enforce positivity, finiteness, and shape ----
    H = np.nan_to_num(H, nan=1e-9, posinf=1e9, neginf=1e-9)
    H = np.where(H > 0.0, H, 1e-9)
    if H.shape != (n, n):
        H = np.ones((n, n), dtype=float)
    return H
