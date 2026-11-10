# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 590.0  # hard wall-clock guard (evaluator kills at 630s)


def _time_left():
    return _DEADLINE - time.time()


def _union_length(x, t=0.0):
    """Exact 1D union length of intervals [x_i, x_i + 1/8].
    Returns (length, gradient w.r.t. x)."""
    n = x.size
    if n == 0:
        return 0.0, np.zeros(0)
    w = 1.0 / 8.0
    lo = x
    hi = x + w
    order = np.argsort(lo)
    lo = lo[order]
    hi = hi[order]
    # active set: leftmost point of the current merged component
    active = np.zeros(n, dtype=bool)
    active[0] = True
    cur = hi[0]
    for i in range(1, n):
        if lo[i] <= cur:
            if hi[i] > cur:
                cur = hi[i]
                active[i] = True
        else:
            active[i] = True
            cur = hi[i]
    total = cur - lo[0]
    grad = np.zeros(n)
    grad[active] = 1.0
    # smooth (soft-max) surrogate: gradient = softmax over hi
    if t > 0.0:
        sm = hi - np.max(hi)
        sm = np.exp(sm / t)
        sm = sm / sm.sum()
        total = np.max(hi) - lo[0] + t * np.log(sm.sum())
        grad = sm
    grad = grad[order]
    return total, grad


def _cost_1d(x, t):
    L, g = _union_length(x, t)
    return L, g


def _cost_3d(x, y, t):
    Lx, gx = _union_length(x, t)
    Ly, gy = _union_length(y, t)
    return Lx * Ly, gx * Ly, gy * Lx


def _mc_volume(positions, n=200000):
    """Exact Monte Carlo estimate per the public task contract (seed 12345)."""
    N = 8
    xk = positions[:, 0]
    yk = positions[:, 1]
    i = (np.arange(64) // N).astype(float)
    j = (np.arange(64) % N).astype(float)
    X0 = xk.min()
    X1 = (xk + (i + 1) / N).max()
    Y0 = yk.min()
    Y1 = (yk + (j + 1) / N).max()
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return 0.0
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n)
    b = rng.uniform(Y0, Y1, n)
    c = rng.uniform(0.0, 1.0, n)
    a2 = a[None, :]
    b2 = b[None, :]
    c2 = c[None, :]
    xl = xk + c2 * i / N
    xr = xl + 1.0 / N
    yl = yk + c2 * j / N
    yr = yl + 1.0 / N
    inside = ((a2 >= xl) & (a2 <= xr) & (b2 >= yl) & (b2 <= yr)).any(axis=0)
    frac = inside.mean()
    return frac * (X1 - X0) * (Y1 - Y0)


def _keich_positions():
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    return np.stack([keich[i_flat], keich[j_flat]], axis=1)


def _optimize_1d(x0, iters):
    """Optimize a single 1D base set to minimize union length."""
    x = x0.copy()
    best = x.copy()
    bestL, _ = _union_length(x, 0.0)
    w = 1.0 / 8.0
    for it in range(iters):
        if _time_left() < 6.0:
            break
        L, g = _union_length(x, 0.0)
        if L < bestL - 1e-15:
            bestL = L
            best = x.copy()
        # L-BFGS-B on exact objective (gradient of union length)
        from scipy.optimize import minimize
        try:
            res = minimize(
                lambda v: _union_length(v, 0.0)[0],
                x,
                jac=lambda v: _union_length(v, 0.0)[1],
                method="L-BFGS-B",
                options={"maxiter": 12, "ftol": 1e-14, "gtol": 1e-10},
            )
            x = res.x
        except Exception:
            pass
        # simulated annealing: accept a worse move with probability exp(-d/T)
        L, g = _union_length(x, 0.0)
        T = max(1e-4, 0.02 * (1.0 - it / iters))
        step = 0.05 * (1.0 - 0.5 * it / iters)
        for _ in range(3):
            if _time_left() < 6.0:
                break
            cand = x + np.random.randn(x.size) * step
            Lc, _ = _union_length(cand, 0.0)
            if Lc <= L or np.random.rand() < np.exp(-(Lc - L) / T):
                x = cand
                L = Lc
                if L < bestL - 1e-15:
                    bestL = L
                    best = x.copy()
    # polish: tiny L-BFGS-B with tight tolerance
    try:
        from scipy.optimize import minimize
        res = minimize(
            lambda v: _union_length(v, 0.0)[0],
            best,
            jac=lambda v: _union_length(v, 0.0)[1],
            method="L-BFGS-B",
            options={"maxiter": 30, "ftol": 1e-16, "gtol": 1e-12},
        )
        if _union_length(res.x, 0.0)[0] < bestL:
            best = res.x
            bestL = _union_length(res.x, 0.0)[0]
    except Exception:
        pass
    return best, bestL


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    base = _keich_positions()
    base_x = base[:, 0]
    base_y = base[:, 1]
    # unique base sets (rows share i for x, j for y)
    ux0 = np.array([base_x[8 * i] for i in range(N)])
    uy0 = np.array([base_y[i] for i in range(N)])
    Lx0, _ = _union_length(ux0, 0.0)
    Ly0, _ = _union_length(uy0, 0.0)
    bestLx, bestLy = Lx0, Ly0
    best_x, best_y = ux0.copy(), uy0.copy()

    # ---- optimize each 1D set, multiple random restarts, under deadline ----
    rng = np.random.default_rng(0)
    restarts = 0
    while _time_left() > 25.0:
        # fresh start: keich + random jitter
        scale = 0.5 if restarts % 2 == 0 else 1.0
        x_start = ux0 + rng.randn(N) * scale * 0.15
        y_start = uy0 + rng.randn(N) * scale * 0.15
        x, Lx = _optimize_1d(x_start, iters=25)
        if Lx < bestLx:
            bestLx, best_x = Lx, x
        y, Ly = _optimize_1d(y_start, iters=25)
        if Ly < bestLy:
            bestLy, best_y = Ly, y
        restarts += 1

    # ---- assemble and validate with the exact Monte Carlo metric ----
    def assemble(bx, by):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        return np.stack([bx[i_flat], by[j_flat]], axis=1)

    cand = assemble(best_x, best_y)
    V = _mc_volume(cand)
    if not (np.isfinite(V) and V > 0):
        cand = base
        V = _mc_volume(cand)
    # keep the better of (optimized, keich) by exact MC volume
    V_base = _mc_volume(base)
    if np.isfinite(V_base) and (not np.isfinite(V) or V_base < V):
        cand = base
    if not np.isfinite(cand).all():
        cand = base
    return cand


# EVOLVE-BLOCK-END
