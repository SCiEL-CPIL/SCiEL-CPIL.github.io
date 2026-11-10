# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).

    Tube (i, j) at height c in [0, 1] is the square
        [x_i + c*i/8, x_i + c*i/8 + 1/8] x [y_j + c*j/8, y_j + c*j/8 + 1/8].

    We minimize the union volume V = integral_0^1 A(c) dc, where A(c) is the
    union area of 64 axis-aligned squares. The x-intervals depend only on i and
    y-intervals only on j, so A(c) can be computed via a fast product structure
    (union of x-intervals per i, union of y-intervals per j). We then optimize
    the 16 free base positions (x_0..x_7, y_0..y_7) to shrink the union.
    """
    deadline = time.time() + 580.0
    N = 8
    w = 1.0 / N
    rng = np.random.default_rng(12345)

    # ---- Fast 1-D union length of N intervals (vectorized over N^2 pairs) ----
    def union_len(starts, widths):
        # starts: (N,), widths: (N,)
        L = np.zeros(N, dtype=np.float64)
        for k in range(N):
            lo = starts[k]
            hi = starts[k] + widths[k]
            # length of [lo, hi] covered by the other intervals
            ov = np.minimum(hi, starts + widths) - np.maximum(lo, starts)
            ov = np.where(ov > 0.0, ov, 0.0)
            L[k] = max(0.0, hi - lo - ov.sum())
        return L.sum()

    # ---- Union area A(c) via product structure (vectorized over c) ----
    def area_at_c(cx, cy, cc):
        # cx, cy: (N,); cc: (M,)
        cc = np.asarray(cc, dtype=np.float64)
        M = cc.shape[0]
        # x-intervals per i: [cx_i + cc*i/N, + w]
        ix = cx[None, :] + cc[:, None] * np.arange(N)[None, :] / N  # (M, N)
        # y-intervals per j
        jy = cy[None, :] + cc[:, None] * np.arange(N)[None, :] / N  # (M, N)

        # union length of x-intervals at each c (vectorized)
        # pair matrix over i
        lo_x = ix[:, :, None]                       # (M, N, 1)
        hi_x = ix[:, :, None] + w                   # (M, N, 1)
        lo_x2 = ix[:, None, :]                      # (M, 1, N)
        hi_x2 = ix[:, None, :] + w                  # (M, 1, N)
        ovx = np.minimum(hi_x, hi_x2) - np.maximum(lo_x, lo_x2)
        ovx = np.where(ovx > 0.0, ovx, 0.0)
        Lx = (hi_x - lo_x) - ovx.sum(axis=2)        # (M, N)
        Lx = np.maximum(Lx, 0.0)
        ux = Lx.sum(axis=1)                          # (M,)

        # same for y
        lo_y = jy[:, :, None]
        hi_y = jy[:, :, None] + w
        lo_y2 = jy[:, None, :]
        hi_y2 = jy[:, None, :] + w
        ovy = np.minimum(hi_y, hi_y2) - np.maximum(lo_y, lo_y2)
        ovy = np.where(ovy > 0.0, ovy, 0.0)
        Ly = (hi_y - lo_y) - ovy.sum(axis=2)
        Ly = np.maximum(Ly, 0.0)
        uy = Ly.sum(axis=1)

        return ux * uy

    # ---- Volume by fixed-sample Monte Carlo (matches evaluator) ----
    def volume_mc(cx, cy):
        X0 = cx.min()
        X1 = (cx + np.arange(N) / N).max()
        Y0 = cy.min()
        Y1 = (cy + np.arange(N) / N).max()
        if (X1 - X0) * (Y1 - Y0) <= 1e-9:
            return None
        a = rng.uniform(X0, X1, 500000)
        b = rng.uniform(Y0, Y1, 500000)
        c = rng.uniform(0.0, 1.0, 500000)
        ix = a[:, None] - (c[:, None] * np.arange(N)[None, :] / N)   # (M, N)
        inx = (ix >= cx[None, :]) & (ix < cx[None, :] + w)
        iy = b[:, None] - (c[:, None] * np.arange(N)[None, :] / N)
        iny = (iy >= cy[None, :]) & (iy < cy[None, :] + w)
        inside = (inx & iny).any(axis=1)
        frac = inside.mean()
        return frac * (X1 - X0) * (Y1 - Y0)

    # ---- Deterministic objective: integral of A(c) (smooth, fast) ----
    CC = np.linspace(0.0, 1.0, 33)  # 33 samples for the smooth objective

    def smooth_obj(z):
        cx = z[:N]
        cy = z[N:2 * N]
        return area_at_c(cx, cy, CC).sum() / len(CC)

    # ---- Seed: Keich binary construction (the score-1.0 baseline) ----
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    def make_positions(cx, cy):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        return np.stack([cx[i_flat], cy[j_flat]], axis=1)

    best_z = np.concatenate([keich, keich]).astype(np.float64)
    best_vol = volume_mc(best_z[:N], best_z[N:2 * N])

    # ---- Multi-start local optimization on the smooth objective ----
    def try_z(z):
        nonlocal best_z, best_vol
        z = np.asarray(z, dtype=np.float64)
        try:
            vol = volume_mc(z[:N], z[N:2 * N])
        except Exception:
            return
        if vol is not None and vol > 0 and (best_vol is None or vol < best_vol):
            best_vol = vol
            best_z = z.copy()

    starts = [best_z.copy()]
    # perturbations around the Keich seed
    for s in range(24):
        step = 0.05 * (1 - s / 30.0)
        starts.append(best_z + rng.normal(0.0, step, size=2 * N))
    # a few from scratch (small random bases)
    for s in range(6):
        starts.append(rng.uniform(-0.5, 0.5, size=2 * N))

    for z0 in starts:
        if time.time() > deadline - 120.0:
            break
        try:
            res = minimize(smooth_obj, z0, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 20000,
                                    "ftol": 1e-12, "gtol": 1e-8})
        except Exception:
            continue
        if time.time() > deadline - 60.0:
            try_z(res.x)
            break
        try_z(res.x)

    # ---- Final polish: a couple of extra L-BFGS-B refinements from the best ----
    if best_vol is not None:
        for _ in range(3):
            if time.time() > deadline - 30.0:
                break
            try:
                res = minimize(smooth_obj, best_z, method="L-BFGS-B",
                               options={"maxiter": 300, "maxfun": 15000,
                                        "ftol": 1e-13, "gtol": 1e-9})
                try_z(res.x)
            except Exception:
                break

    positions = make_positions(best_z[:N], best_z[N:2 * N])

    # ---- Safety: ensure a valid, finite, non-degenerate result ----
    if (not np.all(np.isfinite(positions))) or best_vol is None or best_vol <= 0:
        positions = make_positions(keich, keich)

    return positions


# EVOLVE-BLOCK-END
