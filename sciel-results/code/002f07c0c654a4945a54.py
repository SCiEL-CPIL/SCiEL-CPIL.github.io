# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DL = 600.0
_MARGIN = 15.0


def _deadline_ok():
    return (time.time() - _T0) < (_DL - _MARGIN)


def _keich(N):
    pos = int(round(np.log2(N)))
    j_terms = np.arange(1, pos + 1, dtype=float)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = [int(b) for b in bin(idx)[2:]]
        bits = [0] * (pos - len(bits)) + bits
        out[idx] = np.sum((1.0 - j_terms) * np.array(bits) * (2.0 ** -j_terms) / pos)
    return out


def _keich_positions(N):
    keich = _keich(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _vol1d(p, N):
    # 1D swept area A = sum_c | union of intervals [p_i + c*i/N, p_i + 1/N + c*i/N] |
    c = np.linspace(0.0, 1.0, 1025)
    cc = c[:, None]
    L = p[None, :] + cc * (np.arange(N) / N)
    R = L + 1.0 / N
    a = np.min(L, axis=1)
    b = np.max(R, axis=1)
    inside = (c[:, None] >= L) & (c[:, None] < R)
    cov = np.count_nonzero(inside, axis=1)
    length = np.where(cov > 0, b - a, 0.0)
    return 0.5 * (length[0] + length[-1] + 2.0 * np.sum(length[1:-1])) / 1024.0


def _make_1d(N, seed):
    rng = np.random.default_rng(seed)
    # Centered shear alignment: p_i = -i/(2N)
    # This makes intervals tile [−1/(2N), (N−1)/(2N)] at c=0 and [0, 1] at c=1
    p0 = -np.arange(N, dtype=float) / (2.0 * N)
    # Add small perturbation for different seeds
    p0 = p0 + rng.uniform(-0.01, 0.01, N)
    # Compute bounds: p_i must allow intervals to stay in a reasonable range
    # At c in [0,1], interval i is [p_i + c*i/N, p_i + c*i/N + 1/N]
    # So p_i + 0 >= some_min and p_i + i/N + 1/N <= some_max
    # We want the union to be compact, so bound p_i to [-1, 1]
    lo = np.full(N, -1.0)
    hi = np.full(N, 1.0)
    return p0, lo, hi


def _opt_1d(N, seed):
    p0, lo, hi = _make_1d(N, seed)
    res = minimize(_vol1d, p0, args=(N,), method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12, "maxls": 50})
    p = res.x
    # light polish
    if _deadline_ok():
        r2 = minimize(_vol1d, p, args=(N,), method="L-BFGS-B",
                      bounds=list(zip(lo, hi)),
                      options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-14, "maxls": 50})
        if r2.success:
            p = r2.x
    return p


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    try:
        keich = _keich(N)
        # x: optimized 1D Kakeya needle (minimizes A_x); y: same optimized (minimizes A_y)
        # Volume = A_x * A_y, so optimize both
        p_best = None
        v_best = np.inf
        for seed in (0, 1, 2, 3, 4):
            if not _deadline_ok():
                break
            try:
                p = _opt_1d(N, seed)
                v = _vol1d(p, N)
                if np.isfinite(v) and v < v_best:
                    v_best = v
                    p_best = p.copy()
            except Exception:
                pass
        if p_best is None:
            p_best = keich.copy()

        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        base_x = p_best[i_grid.flatten()]
        base_y = p_best[j_grid.flatten()]
        positions = np.stack([base_x, base_y], axis=1)

        if not (np.all(np.isfinite(positions)) and positions.shape == (N * N, 2)):
            positions = _keich_positions(N)
        return positions
    except Exception:
        return _keich_positions(N)


# EVOLVE-BLOCK-END
