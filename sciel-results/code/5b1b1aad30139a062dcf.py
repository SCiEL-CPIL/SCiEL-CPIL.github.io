# EVOLVE-BLOCK-START
import time
import numpy as np


def _union_1d_length(intervals):
    """Total length of the union of closed intervals (n,2) -> scalar."""
    if len(intervals) == 0:
        return 0.0
    order = np.argsort(intervals[:, 0])
    s = intervals[order, 0]
    e = intervals[order, 1]
    total = 0.0
    cur_s = s[0]
    cur_e = e[0]
    for k in range(1, len(s)):
        if s[k] <= cur_e:
            if e[k] > cur_e:
                cur_e = e[k]
        else:
            total += cur_e - cur_s
            cur_s = s[k]
            cur_e = e[k]
    total += cur_e - cur_s
    return total


def _union_volume(base, n_c=48):
    """Analytic union volume of the 64 sheared tubes.

    base: (64,2) array of base positions (x_k, y_k).
    Returns the volume of the union over c in [0,1].

    At each height c the cross-section is the union of 64 axis-aligned
    squares of side w. Its area equals (union length in x) * (union
    length in y) only when the union is a rectangle; in general the
    area is the 2D union. We compute the exact 2D union area at each c
    via coordinate compression (the union of axis-aligned rectangles
    is piecewise-constant between sorted x and y edges).
    """
    N = 8
    k = np.arange(64)
    i = (k // N).astype(np.float64)
    j = (k % N).astype(np.float64)
    w = 1.0 / N

    x0_all = base[:, 0]
    y0_all = base[:, 1]
    # bounding box in x and y over all c in [0,1]
    x1_all = x0_all + i / N + w
    y1_all = y0_all + j / N + w
    X0 = x0_all.min()
    X1 = x1_all.max()
    Y0 = y0_all.min()
    Y1 = y1_all.max()
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 0.0

    # Use a moderate number of c samples; the integrand is piecewise
    # linear in c (each edge is affine in c), so the area is piecewise
    # linear between consecutive critical c values. We sample densely
    # enough to capture the structure.
    c_vals = np.linspace(0.0, 1.0, n_c)
    total_area = 0.0
    for c in c_vals:
        sx0 = x0_all + c * i / N
        sx1 = sx0 + w
        sy0 = y0_all + c * j / N
        sy1 = sy0 + w
        xs = np.unique(np.concatenate([sx0, sx1]))
        ys = np.unique(np.concatenate([sy0, sy1]))
        nx = len(xs) - 1
        ny = len(ys) - 1
        if nx <= 0 or ny <= 0:
            continue
        cx = 0.5 * (xs[:-1] + xs[1:])
        cy = 0.5 * (ys[:-1] + ys[1:])
        CX, CY = np.meshgrid(cx, cy, indexing="xy")  # (ny, nx)
        CXf = CX.ravel()
        CYf = CY.ravel()
        inside = ((CXf[:, None] >= sx0[None, :]) &
                  (CXf[:, None] < sx1[None, :]) &
                  (CYf[:, None] >= sy0[None, :]) &
                  (CYf[:, None] < sy1[None, :])).any(axis=1)
        covered = inside.reshape(ny, nx)
        dx = np.diff(xs)
        dy = np.diff(ys)
        cell_area = np.outer(dy, dx)
        total_area += (covered * cell_area).sum()
    vol = total_area * (1.0 / n_c)
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

    from scipy.optimize import minimize

    def obj(v):
        b = v.reshape(64, 2)
        vvol = _union_volume(b, n_c=32)
        return vvol

    def obj_high(v):
        b = v.reshape(64, 2)
        vvol = _union_volume(b, n_c=64)
        return vvol

    best = base.copy()
    best_val = _union_volume(best, n_c=48)

    # Diverse multi-start: centered fan plus perturbations at several scales
    rng = np.random.default_rng(0)
    starts = [base.copy()]
    for scale in [0.02, 0.05, 0.10, 0.15, 0.20]:
        pert = base + rng.normal(0, scale, size=base.shape)
        starts.append(pert)

    for init in starts:
        if time.time() > DEADLINE:
            break
        # Phase 1: L-BFGS-B for fast smooth convergence
        try:
            res = minimize(obj, init.ravel(), method="L-BFGS-B",
                           options=dict(maxiter=200, maxfun=4000, ftol=1e-12, gtol=1e-10))
            cand = res.x.reshape(64, 2)
            if np.all(np.isfinite(cand)):
                cv = _union_volume(cand, n_c=48)
                if cv < best_val:
                    best_val = cv
                    best = cand
        except Exception:
            pass
        # Phase 2: Powell for robust polishing near nonsmooth minimum
        if time.time() > DEADLINE - 120:
            break
        try:
            res2 = minimize(obj, best.ravel(), method="Powell",
                            options=dict(maxiter=100, maxfun=5000, ftol=1e-12))
            cand2 = res2.x.reshape(64, 2)
            if np.all(np.isfinite(cand2)):
                cv2 = _union_volume(cand2, n_c=48)
                if cv2 < best_val:
                    best_val = cv2
                    best = cand2
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < DEADLINE - 60:
        try:
            res = minimize(obj_high, best.ravel(), method="L-BFGS-B",
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
        # One more Powell polish at high precision
        if time.time() < DEADLINE - 30:
            try:
                res2 = minimize(obj_high, best.ravel(), method="Powell",
                                options=dict(maxiter=80, maxfun=4000, ftol=1e-14))
                cand2 = res2.x.reshape(64, 2)
                if np.all(np.isfinite(cand2)):
                    cv2 = _union_volume(cand2, n_c=64)
                    if cv2 < best_val:
                        best_val = cv2
                        best = cand2
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
