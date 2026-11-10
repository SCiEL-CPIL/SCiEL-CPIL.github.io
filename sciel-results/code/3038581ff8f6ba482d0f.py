# EVOLVE-BLOCK-START
import time
import numpy as np


def _union_volume(base, n_c=48):
    """Analytic union volume of the 64 sheared tubes.

    base: (64,2) array of base positions (x_k, y_k).
    Returns the volume of the union over c in [0,1].
    """
    N = 8
    k = np.arange(64)
    i = (k // N).astype(np.float64)
    j = (k % N).astype(np.float64)
    w = 1.0 / N

    # bounding box in x and y over all c in [0,1]
    x0_all = base[:, 0]
    x1_all = base[:, 0] + i / N + w
    y0_all = base[:, 1]
    y1_all = base[:, 1] + j / N + w
    X0 = x0_all.min()
    X1 = x1_all.max()
    Y0 = y0_all.min()
    Y1 = y1_all.max()
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 0.0

    c_vals = np.linspace(0.0, 1.0, n_c)
    total_area = 0.0
    for c in c_vals:
        # square k at height c
        sx0 = x0_all + c * i / N
        sx1 = sx0 + w
        sy0 = y0_all + c * j / N
        sy1 = sy0 + w
        # coordinate compression
        xs = np.unique(np.concatenate([sx0, sx1]))
        ys = np.unique(np.concatenate([sy0, sy1]))
        nx = len(xs) - 1
        ny = len(ys) - 1
        if nx <= 0 or ny <= 0:
            continue
        # cell centers
        cx = 0.5 * (xs[:-1] + xs[1:])
        cy = 0.5 * (ys[:-1] + ys[1:])
        # coverage: for each cell, is it inside any square?
        # vectorize: build (nx,ny) grid of centers, test against 64 squares
        CX, CY = np.meshgrid(cx, cy, indexing="xy")  # shape (ny, nx)
        CXf = CX.ravel()
        CYf = CY.ravel()
        # squares as (64,4): sx0,sx1,sy0,sy1
        inside = ((CXf[:, None] >= sx0[None, :]) &
                  (CXf[:, None] < sx1[None, :]) &
                  (CYf[:, None] >= sy0[None, :]) &
                  (CYf[:, None] < sy1[None, :])).any(axis=1)
        covered = inside.reshape(ny, nx)
        # area of each cell
        dx = np.diff(xs)
        dy = np.diff(ys)
        cell_area = np.outer(dy, dx)  # (ny, nx)
        total_area += (covered * cell_area).sum()
    # trapezoidal integration over c
    vol = np.trapz(total_area, c_vals) if False else total_area * (1.0 / n_c)
    return vol


def _mc_volume(base, seed=12345):
    """The exact Monte Carlo estimate used by the evaluator."""
    N = 8
    k = np.arange(64)
    i = (k // N).astype(np.float64)
    j = (k % N).astype(np.float64)
    w = 1.0 / N
    x0_all = base[:, 0]
    x1_all = base[:, 0] + (i + 1) / N  # note: X1 uses (i+1)/8
    y0_all = base[:, 1]
    y1_all = base[:, 1] + (j + 1) / N
    X0 = x0_all.min()
    X1 = x1_all.max()
    Y0 = y0_all.min()
    Y1 = y1_all.max()
    if not (np.isfinite(X0) and np.isfinite(X1) and np.isfinite(Y0) and np.isfinite(Y1)):
        return 0.0
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0.0, 1.0, 500000)
    # inside any tube?
    # tube k: a in [x_k + c*i/8, x_k + c*i/8 + 1/8] and b in [y_k + c*j/8, y_k + c*j/8 + 1/8]
    # vectorize over 64 tubes
    ax0 = x0_all[None, :] + c[:, None] * i[None, :] / N
    ax1 = ax0 + w
    by0 = y0_all[None, :] + c[:, None] * j[None, :] / N
    by1 = by0 + w
    inside = ((a[:, None] >= ax0) & (a[:, None] < ax1) &
              (b[:, None] >= by0) & (b[:, None] < by1)).any(axis=1)
    frac = inside.mean()
    return frac * (X1 - X0) * (Y1 - Y0)


def _centered_fan_init():
    N = 8
    k = np.arange(64)
    i = (k // N).astype(np.float64)
    j = (k % N).astype(np.float64)
    c_star = 0.5
    x_star = 0.0
    y_star = 0.0
    xk = x_star - 0.5 / N - c_star * i / N
    yk = y_star - 0.5 / N - c_star * j / N
    return np.stack([xk, yk], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    """Return 64 base positions minimizing the 3D union volume of the sheared tubes."""
    t_start = time.time()
    DEADLINE = 600.0 - 30.0  # safety margin

    base = _centered_fan_init()

    # Local polish with L-BFGS-B on the analytic volume.
    from scipy.optimize import minimize

    def obj(v):
        b = v.reshape(64, 2)
        vvol = _union_volume(b, n_c=32)
        return vvol

    best = base.copy()
    best_val = _union_volume(best, n_c=48)

    # multi-start: a few perturbations
    rng = np.random.default_rng(0)
    starts = [base.copy()]
    for s in range(4):
        pert = base + rng.normal(0, 0.05, size=base.shape)
        starts.append(pert)

    for init in starts:
        if time.time() > DEADLINE:
            break
        try:
            res = minimize(obj, init.ravel(), method="L-BFGS-B",
                           options=dict(maxiter=200, maxfun=4000, ftol=1e-12, gtol=1e-10))
            if res.success or res.fun < best_val:
                cand = res.x.reshape(64, 2)
                if np.all(np.isfinite(cand)):
                    cv = _union_volume(cand, n_c=48)
                    if cv < best_val:
                        best_val = cv
                        best = cand
        except Exception:
            pass
        if time.time() > DEADLINE:
            break

    # Final high-precision polish from the best point
    if time.time() < DEADLINE - 60:
        try:
            res = minimize(obj, best.ravel(), method="L-BFGS-B",
                           options=dict(maxiter=300, maxfun=6000, ftol=1e-14, gtol=1e-12))
            if res.fun < best_val:
                cand = res.x.reshape(64, 2)
                if np.all(np.isfinite(cand)):
                    cv = _union_volume(cand, n_c=64)
                    if cv < best_val:
                        best_val = cv
                        best = cand
        except Exception:
            pass

    # Verify with the exact MC estimator; if invalid, fall back to seed.
    mc = _mc_volume(best)
    if (not np.isfinite(mc)) or mc <= 0:
        # fallback to the original Keich construction
        N = 8
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        base_x = keich[i_grid.flatten()]
        base_y = keich[j_grid.flatten()]
        best = np.stack([base_x, base_y], axis=1)

    return best


# EVOLVE-BLOCK-END
