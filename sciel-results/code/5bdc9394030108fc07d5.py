# EVOLVE-BLOCK-START
import time
import numpy as np
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
_DELTA = _CCW - _CW  # (D, N)
_BASE_CW = _CW.sum(axis=0)  # (N,)


def _maxload(x):
    loads = _BASE_CW + x @ _DELTA
    return float(loads.max())


def _milp_solve(deadline):
    D, N = _D, _N
    c = np.zeros(D + 1)
    c[D] = 1.0

    A_ub = np.zeros((N, D + 1))
    A_ub[:, :D] = _DELTA.T
    A_ub[:, D] = -1.0
    b_ub = -_BASE_CW

    bounds = [(0, 1)] * D + [(0, None)]
    integrality = [1] * D + [0]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      integrality=integrality, method='highs',
                      options={'time_limit': max(1.0, deadline - time.time())})
        if res.success:
            x = np.where(res.x[:D] > 0.5, 1.0, 0.0)
            val = _maxload(x)
            return x, val
    except Exception:
        pass
    return None, float('inf')


def _brute_force_optimal():
    """Exhaustive search over all 2^15 = 32768 assignments. Exact."""
    best_x = None
    best_val = float('inf')
    D = _D
    for mask in range(1 << D):
        x = np.zeros(D, dtype=np.float64)
        for i in range(D):
            if (mask >> i) & 1:
                x[i] = 1.0
        v = _maxload(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    return best_x, best_val


def _hill_climb(x, deadline, max_iter=50000):
    x = x.copy()
    cur = _maxload(x)
    best = cur
    bestx = x.copy()
    n = _D
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = np.random.randint(n)
        x[i] = 1.0 - x[i]
        cand = _maxload(x)
        if cand < cur:
            cur = cand
            if cur < best:
                best = cur
                bestx = x.copy()
        else:
            x[i] = 1.0 - x[i]
    return bestx, best


def _sa_polish(x, deadline, t0=1.5, t1=0.001, iters=50000):
    x = x.copy()
    cur = _maxload(x)
    best = cur
    bestx = x.copy()
    for it in range(iters):
        if time.time() > deadline:
            break
        frac = it / max(iters, 1)
        t = t0 * (t1 / t0) ** frac
        i = np.random.randint(_D)
        x[i] = 1.0 - x[i]
        cand = _maxload(x)
        d = cand - cur
        if d < 0 or np.random.random() < np.exp(-d / max(t, 1e-9)):
            cur = cand
            if cur < best:
                best = cur
                bestx = x.copy()
        else:
            x[i] = 1.0 - x[i]
    return bestx, best


def _systematic_search(x, deadline):
    """Exhaustive 1-flip and 2-flip local search."""
    x = x.copy()
    cur = _maxload(x)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-flips
        for i in range(_D):
            if time.time() > deadline:
                return x, cur
            x[i] = 1.0 - x[i]
            cand = _maxload(x)
            if cand < cur:
                cur = cand
                improved = True
            else:
                x[i] = 1.0 - x[i]
        # 2-flips
        for i in range(_D):
            for j in range(i + 1, _D):
                if time.time() > deadline:
                    return x, cur
                x[i] = 1.0 - x[i]
                x[j] = 1.0 - x[j]
                cand = _maxload(x)
                if cand < cur:
                    cur = cand
                    improved = True
                else:
                    x[i] = 1.0 - x[i]
                    x[j] = 1.0 - x[j]
    return x, cur


def ring_routing() -> np.ndarray:
    np.random.seed(42)
    start = time.time()
    deadline = start + 570.0  # safety margin before 630s kill

    # Phase 1: Exact brute force (2^15 = 32768 evaluations, ~1-2s)
    bf_x, bf_val = _brute_force_optimal()

    # Phase 2: MILP as cross-check / potential improvement
    milp_x, milp_val = _milp_solve(deadline)

    best = bf_val
    bestx = bf_x.copy()
    if milp_x is not None and milp_val < best:
        best = milp_val
        bestx = milp_x.copy()

    # Phase 3: Systematic local search (1-flip + 2-flip) from best
    if time.time() < deadline - 30.0:
        bestx, best = _systematic_search(bestx, deadline - 30.0)

    # Phase 4: Simulated annealing polish
    while time.time() < deadline - 10.0:
        x = bestx.copy()
        x, s = _sa_polish(x, deadline - 10.0)
        if s < best:
            best = s
            bestx = x.copy()
        if time.time() > deadline - 20.0:
            break

    # Phase 5: Final systematic pass
    if time.time() < deadline - 1.0:
        bestx, best = _systematic_search(bestx, deadline - 1.0)

    return bestx


# EVOLVE-BLOCK-END
