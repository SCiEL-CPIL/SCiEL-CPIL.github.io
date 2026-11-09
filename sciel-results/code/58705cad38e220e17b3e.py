# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

_N = 10
_DEMANDS = (
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
)
_D = len(_DEMANDS)


def _build():
    cw = np.zeros((_D, _N), dtype=np.float64)
    ccw = np.zeros((_D, _N), dtype=np.float64)
    for i, (s, t, w) in enumerate(_DEMANDS):
        if t > s:
            cw[i, s:t] = w
        else:
            cw[i, s:] = w
            cw[i, :t] = w
        ccw[i] = w - cw[i]
    return cw, ccw


_CW, _CCW = _build()
# Precompute delta vectors: flipping demand i changes load by (target - source)
# If currently CW and we flip to CCW: delta = CCW[i] - CW[i]
# If currently CCW and we flip to CW: delta = CW[i] - CCW[i]
# We'll compute both and select based on current state.
_DELTA_CW_TO_CCW = _CCW - _CW  # shape (D, N)
_DELTA_CCW_TO_CW = _CW - _CCW  # shape (D, N), = -_DELTA_CW_TO_CCW


def _maxload_fast(x, cw, ccw):
    """Compute max edge load given binary assignment x (0=cw, 1=ccw)."""
    loads = np.zeros(_N, dtype=np.float64)
    # Vectorized: for each demand, add cw[i] if x[i]==0 else ccw[i]
    # Use np.where per edge? Better: loads = x @ ccw.T + (1-x) @ cw.T  -> but x is (D,)
    # Actually: loads[e] = sum_i x[i]*ccw[i,e] + (1-x[i])*cw[i,e]
    #           = sum_i cw[i,e] + sum_i x[i]*(ccw[i,e]-cw[i,e])
    #           = cw.sum(axis=0) + x @ _DELTA_CW_TO_CCW
    base = cw.sum(axis=0)
    loads = base + x @ _DELTA_CW_TO_CCW
    return float(loads.max())


def _loads_vector(x, cw):
    base = cw.sum(axis=0)
    return base + x @ _DELTA_CW_TO_CCW


def _milp_solve(cw, ccw, deadline):
    """Solve the MILP: min t s.t. edge loads <= t, x_i in {0,1}.
    Returns (bestx, best_val) or (None, inf) on failure/timeout.
    """
    D, N = _D, _N
    # Decision variables: x_0..x_{D-1} (binary), t (continuous)
    # Objective: minimize t  -> c = [0]*D + [1]
    c = np.zeros(D + 1)
    c[D] = 1.0

    # Inequality constraints: A_ub @ vars <= b_ub
    # For each edge e: sum_i x_i * delta_cw2ccw[i,e] + base_cw[e] <= t
    #   => sum_i delta[i,e]*x_i - t <= -base_cw[e]
    base_cw = cw.sum(axis=0)  # (N,)
    delta = _DELTA_CW_TO_CCW  # (D, N)

    A_ub = np.zeros((N, D + 1))
    A_ub[:, :D] = delta.T  # (N, D)
    A_ub[:, D] = -1.0
    b_ub = -base_cw

    # Bounds: x_i in [0,1], t in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]

    # Integrality: x binary, t continuous
    integrality = [1] * D + [0]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      integrality=integrality, method='highs',
                      options={'time_limit': max(1.0, deadline - time.time())})
        if res.success:
            x = res.x[:D].astype(np.float64)
            # Round to nearest 0/1 (should already be near-integer)
            x = np.where(x > 0.5, 1.0, 0.0)
            val = _maxload_fast(x, cw, ccw)
            return x, val
    except Exception:
        pass
    return None, float('inf')


def _hill_climb(x, cw, ccw, max_iter, deadline, use_double=True, use_triple=False):
    """Hill climbing with 1-flip, optionally 2-flip and 3-flip neighborhoods."""
    x = x.copy()
    cur = _maxload_fast(x, cw, ccw)
    best = cur
    bestx = x.copy()
    n = _D

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Choose neighborhood size
        r = random.random()
        if use_triple and r < 0.1:
            k = 3
        elif use_double and r < 0.4:
            k = 2
        else:
            k = 1

        if k == 1:
            i = random.randrange(n)
            x[i] = 1.0 - x[i]
            cand = _maxload_fast(x, cw, ccw)
            if cand < cur:
                cur = cand
                if cur < best:
                    best = cur
                    bestx = x.copy()
            else:
                x[i] = 1.0 - x[i]
        elif k == 2:
            i, j = random.sample(range(n), 2)
            x[i] = 1.0 - x[i]
            x[j] = 1.0 - x[j]
            cand = _maxload_fast(x, cw, ccw)
            if cand < cur:
                cur = cand
                if cur < best:
                    best = cur
                    bestx = x.copy()
            else:
                x[i] = 1.0 - x[i]
                x[j] = 1.0 - x[j]
        elif k == 3:
            i, j, m = random.sample(range(n), 3)
            x[i] = 1.0 - x[i]
            x[j] = 1.0 - x[j]
            x[m] = 1.0 - x[m]
            cand = _maxload_fast(x, cw, ccw)
            if cand < cur:
                cur = cand
                if cur < best:
                    best = cur
                    bestx = x.copy()
            else:
                x[i] = 1.0 - x[i]
                x[j] = 1.0 - x[j]
                x[m] = 1.0 - x[m]

    return bestx, best


def _simulated_anneal(x, cw, ccw, t0, t1, iters, deadline):
    x = x.copy()
    cur = _maxload_fast(x, cw, ccw)
    best = cur
    bestx = x.copy()

    for it in range(iters):
        if time.time() > deadline:
            break
        frac = it / max(iters, 1)
        t = t0 * (t1 / t0) ** frac
        i = random.randrange(_D)
        x[i] = 1.0 - x[i]
        cand = _maxload_fast(x, cw, ccw)
        delta = cand - cur
        if delta < 0 or random.random() < np.exp(-delta / max(t, 1e-9)):
            cur = cand
            if cur < best:
                best = cur
                bestx = x.copy()
        else:
            x[i] = 1.0 - x[i]

    return bestx, best


def _systematic_local_search(x, cw, ccw, deadline):
    """Try all single flips, then all pairs, accept improvements."""
    x = x.copy()
    cur = _maxload_fast(x, cw, ccw)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for i in range(_D):
            if time.time() > deadline:
                return x, cur
            x[i] = 1.0 - x[i]
            cand = _maxload_fast(x, cw, ccw)
            if cand < cur:
                cur = cand
                improved = True
            else:
                x[i] = 1.0 - x[i]
        # Pair flips
        for i in range(_D):
            for j in range(i + 1, _D):
                if time.time() > deadline:
                    return x, cur
                x[i] = 1.0 - x[i]
                x[j] = 1.0 - x[j]
                cand = _maxload_fast(x, cw, ccw)
                if cand < cur:
                    cur = cand
                    improved = True
                else:
                    x[i] = 1.0 - x[i]
                    x[j] = 1.0 - x[j]
    return x, cur


def ring_routing() -> np.ndarray:
    random.seed(42)
    np.random.seed(42)
    start = time.time()
    deadline = start + 580.0  # safety margin before 630s kill

    cw, ccw = _CW, _CCW

    # Phase 0: Try MILP for global optimum (fast for 15 vars)
    milp_x, milp_val = _milp_solve(cw, ccw, deadline)

    # Phase 1: Multi-start hill climbing with double-flip
    bestx = np.zeros(_D, dtype=np.float64)
    best = _maxload_fast(bestx, cw, ccw)

    # If MILP succeeded, use it as initial best
    if milp_x is not None and milp_val < best:
        best = milp_val
        bestx = milp_x.copy()

    # Start from all-CW
    x0 = np.zeros(_D)
    x0, s0 = _hill_climb(x0, cw, ccw, 50000, deadline, use_double=True, use_triple=False)
    if s0 < best:
        best = s0
        bestx = x0.copy()

    # Also start from MILP solution if available
    if milp_x is not None:
        x1 = milp_x.copy()
        x1, s1 = _hill_climb(x1, cw, ccw, 50000, deadline, use_double=True, use_triple=False)
        if s1 < best:
            best = s1
            bestx = x1.copy()

    # Random starts
    n_starts = 0
    while time.time() < deadline - 60.0:
        x = (np.random.rand(_D) < 0.5).astype(np.float64)
        x, s = _hill_climb(x, cw, ccw, 30000, deadline, use_double=True, use_triple=False)
        if s < best:
            best = s
            bestx = x.copy()
            n_starts += 1
        if time.time() > deadline - 120.0:
            break

    # Phase 2: Simulated annealing polishing
    while time.time() < deadline - 30.0:
        x = bestx.copy()
        x, s = _simulated_anneal(x, cw, ccw, 2.0, 0.001, 50000, deadline - 30.0)
        if s < best:
            best = s
            bestx = x.copy()
        if time.time() > deadline - 60.0:
            break

    # Phase 3: Systematic local search (all 1-flips and 2-flips)
    if time.time() < deadline - 5.0:
        bestx, best = _systematic_local_search(bestx, cw, ccw, deadline - 5.0)

    # Phase 4: Final hill climb with triple-flip
    if time.time() < deadline - 1.0:
        bestx, best = _hill_climb(bestx, cw, ccw, 100000, deadline - 1.0, use_double=True, use_triple=True)

    return bestx


# EVOLVE-BLOCK-END
