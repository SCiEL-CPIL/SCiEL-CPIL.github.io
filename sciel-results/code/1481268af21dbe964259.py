# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _keich_init(N):
    """Binary Keich construction for 1-D positions."""
    k = int(np.log2(N))
    vals = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = np.array([int(d) for d in bin(idx)[2:].zfill(k)])
        weights = np.array([(1 - j) * (2.0 ** -j) for j in range(1, k + 1)])
        vals[idx] = np.dot(bits, weights) / k
    return vals


def _union1d(x, N, n_c=300):
    """Estimate the 1-D union length (averaged over c in [0,1]) for intervals of width 1/N
    at positions x[i] + c*i/N.  Uses sorted-interval merging on a grid of c values."""
    w = 1.0 / N
    idx = np.arange(N, dtype=np.float64)
    total = 0.0
    for c in np.linspace(0.0, 1.0, n_c):
        starts = x + c * idx / N
        order = np.argsort(starts)
        s = starts[order]
        e = s + w
        # vectorised merge
        overlaps = np.diff(s) < w
        # number of "new" segments = 1 + number of times a gap >= w appears
        new_segs = 1 + int(np.sum(~overlaps))
        # total union length = new_segs * w (since each merged block has width >= w and
        # the intervals are all width w and overlapping within a block)
        # More precisely, union length = max(e) - min(s) - sum of gaps
        gaps = np.maximum(0.0, s[1:] - e[:-1])
        union_len = (e[-1] - s[0]) - np.sum(gaps)
        total += union_len
    return total / n_c


def _optimize_1d(N, x0, budget):
    """Optimize 1-D positions with differential evolution, then a local polish."""
    t0 = time.time()
    deadline = t0 + budget

    bounds = [(x0.min() - 0.5, x0.max() + 0.5) for _ in range(N)]
    bounds = [(-1.0, 2.0)] * N  # generous bounds

    def obj(x):
        return _union1d(x, N)

    best_x = x0.copy()
    best_val = obj(best_x)

    # Differential evolution with time budget
    if time.time() < deadline - 5:
        try:
            res = differential_evolution(
                obj, bounds, seed=12345, maxiter=200, tol=1e-10,
                popsize=15, init='sobol', polish=False, updating='deferred'
            )
            if res.fun < best_val:
                best_x = res.x.copy()
                best_val = res.fun
        except Exception:
            pass

    # Local polish with Nelder-Mead
    if time.time() < deadline - 2:
        from scipy.optimize import minimize
        try:
            res2 = minimize(obj, best_x, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_val:
                best_x = res2.x.copy()
                best_val = res2.fun
        except Exception:
            pass

    return best_x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    x0 = _keich_init(N)

    total_budget = 600.0
    safety = 30.0
    per_dim = (total_budget - safety) / 2.0

    t0 = time.time()
    x_opt = _optimize_1d(N, x0, per_dim)

    # Use the same optimized positions for y (symmetric problem)
    y_opt = x_opt.copy()

    i_idx = np.repeat(np.arange(N, dtype=np.int64), N)
    j_idx = np.tile(np.arange(N, dtype=np.int64), N)

    base_x = x_opt[i_idx].astype(np.float64)
    base_y = y_opt[j_idx].astype(np.float64)

    positions = np.column_stack([base_x, base_y])

    return positions


# EVOLVE-BLOCK-END
