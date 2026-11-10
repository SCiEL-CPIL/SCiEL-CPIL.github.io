# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length(intervals):
    """Compute total length of union of 1-D intervals."""
    if len(intervals) == 0:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def _exact_volume(pos_x, pos_y, N=8):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_samples = np.linspace(0, 1, 300)
    vol = 0.0
    invN = 1.0 / N
    for c in c_samples:
        x_ints = [(pos_x[i] + c * i * invN, pos_x[i] + c * i * invN + invN) for i in range(N)]
        y_ints = [(pos_y[j] + c * j * invN, pos_y[j] + c * j * invN + invN) for j in range(N)]
        vol += _union_length(x_ints) * _union_length(y_ints)
    return vol * (1.0 / 300)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0

    # Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N)

    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume(params[:N], params[N:], N)

    options = {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
    rng = np.random.default_rng(12345)

    n_restarts = 0
    while time.time() < deadline - 60:
        scale = max(0.005, 0.3 * best_vol ** 0.5)
        init = np.concatenate([best_x, best_y]) + rng.normal(0, scale, 2 * N)
        result = minimize(objective, init, method='Nelder-Mead', options=options)
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
        n_restarts += 1

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
