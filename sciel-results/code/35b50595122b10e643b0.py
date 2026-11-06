"""Antireflection coating design via smooth multi-start local optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# ---- physics: characteristic-matrix reflectance ----
def _reflectance_stack(thetas, n_layers, n_sub, n_amb=1.0):
    """Mean normal-incidence reflectance (%) over wavelengths in thetas (radians phase).
    thetas: (N, n_layers) optical thicknesses (n*d) per wavelength."""
    n = n_layers
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    if n == 0:
        rr = ((n_amb - n_sub) / (n_amb + n_sub)) ** 2
        return 100.0 * float(np.mean(rr))
    if n == 1:
        t = thetas
        B = n_sub * cos[:, 0] - 1j * n_sub * sin[:, 0]
        A = n_amb * cos[:, 0] - 1j * n_amb * sin[:, 0]
        r = (A - B) / (A + B)
        return 100.0 * float(np.mean(np.abs(r) ** 2))
    # general
    B = np.empty((thetas.shape[0], 2), dtype=complex)
    A = np.empty((thetas.shape[0], 2), dtype=complex)
    B[:, 0] = n_sub * cos[:, -1]
    B[:, 1] = -1j * n_sub * sin[:, -1]
    A[:, 0] = cos[:, -1]
    A[:, 1] = -1j * (sin[:, -1] / n_sub)
    for k in range(n - 2, -1, -1):
        c = cos[:, k]
        s = sin[:, k]
        nk = n_layers and None  # placeholder
    return 0.0  # replaced below


def design(problem):
    n_amb = float(problem.get("ambient_index", 1.0))
    n_sub = float(problem.get("substrate_index", 1.52))
    nL = float(problem["materials"]["L"])
    nH = float(problem["materials"]["H"])
    w0, w1 = problem["band_nm"]
    max_layers = int(problem.get("max_layers", 20))
    max_total = float(problem.get("max_total_nm", 1000))
    max_layer = float(problem.get("max_layer_nm", 400))
    n_w = 401

    t_start = time.time()
    DEADLINE = 142.0

    # fixed layer sequence: alternating L/H from the air side, max_layers
    seq = ['L' if i % 2 == 0 else 'H' for i in range(max_layers)]
    n = max_layers
    n_of = np.array([nL if m == 'L' else nH for m in seq], dtype=float)

    # wavelength grid (401 points)
    wv = np.linspace(w0, w1, n_w)
    k = 2.0 * np.pi / wv  # (n_w,)

    # precompute per-layer, per-wavelength optical-thickness coefficients
    # d_opt = n * d ; theta = k * d_opt = k * n * d  -> coeff[:,j] = k * n_of[j]
    coeff = k[:, None] * n_of[None, :]  # (n_w, n)

    # build the (n-1) fixed 2x2 layer matrices (for layers 0..n-2), per wavelength
    # M_j = [[cos, i sin/n_j],[i n_j sin, cos]]
    # Precompute cos/sin per (wavelength, layer) as function of d via theta.
    # We'll compute cos/sin on the fly but cache the 1/n_j factor.
    inv_n = 1.0 / n_of  # (n,)
    # fixed matrices for j in 0..n-2
    # We store per-layer arrays of cos/sin computed from theta.

    def objective(d):
        d = np.asarray(d, dtype=float)
        theta = coeff * d[None, :]  # (n_w, n)
        cos = np.cos(theta)
        sin = np.sin(theta)
        # propagate from last layer (n-1) down to layer 0
        B0 = n_sub * cos[:, -1]
        B1 = -1j * n_sub * sin[:, -1]
        A0 = cos[:, -1]
        A1 = -1j * (sin[:, -1] * inv_n[-1])
        for j in range(n - 2, -1, -1):
            cj = cos[:, j]
            sj = sin[:, j]
            # M_j * [B0,B1]
            nB = cj * B0 + 1j * (sj * inv_n[j]) * B1
            nA = cj * A0 + 1j * (sj * n_of[j]) * A1
            B0, B1 = nB, nA
            A0, A1 = nA, nA  # careful: A1 overwritten; recompute below
        # recompute A correctly (fix the bug above)
        # redo propagation cleanly
        B0 = n_sub * cos[:, -1]
        B1 = -1j * n_sub * sin[:, -1]
        A0 = cos[:, -1]
        A1 = -1j * (sin[:, -1] * inv_n[-1])
        for j in range(n - 2, -1, -1):
            cj = cos[:, j]
            sj = sin[:, j]
            nB = cj * B0 + 1j * (sj * inv_n[j]) * B1
            nA = cj * A0 + 1j * (sj * n_of[j]) * A1
            B0, B1 = nB, nA
            A0, A1 = nA, nA
        # still buggy for A1; do a clean vectorized build instead
        # (clean rebuild)
        B = np.stack([n_sub * cos[:, -1], -1j * n_sub * sin[:, -1]], axis=1)
        A = np.stack([cos[:, -1], -1j * sin[:, -1] * inv_n[-1]], axis=1)
        for j in range(n - 2, -1, -1):
            cj = cos[:, j]
            sj = sin[:, j]
            # M_j = [[cj, 1j*sj/n_j],[1j*n_j*sj, cj]]
            B = cj[:, None] * B + (1j * sj[:, None] * inv_n[j]) * B[:, [1]]
            # wrong mixing; use explicit
        # Final clean explicit propagation:
        Bm = np.empty((n_w, 2), dtype=complex)
        Am = np.empty((n_w, 2), dtype=complex)
        Bm[:, 0] = n_sub * cos[:, -1]
        Bm[:, 1] = -1j * n_sub * sin[:, -1]
        Am[:, 0] = cos[:, -1]
        Am[:, 1] = -1j * sin[:, -1] * inv_n[-1]
        for j in range(n - 2, -1, -1):
            cj = cos[:, j]
            sj = sin[:, j]
            m00 = cj
            m01 = 1j * sj * inv_n[j]
            m10 = 1j * sj * n_of[j]
            m11 = cj
            nB0 = m00 * Bm[:, 0] + m01 * Bm[:, 1]
            nB1 = m10 * Bm[:, 0] + m11 * Bm[:, 1]
            nA0 = m00 * Am[:, 0] + m01 * Am[:, 1]
            nA1 = m10 * Am[:, 0] + m11 * Am[:, 1]
            Bm[:, 0] = nB0
            Bm[:, 1] = nB1
            Am[:, 0] = nA0
            Am[:, 1] = nA1
        r = (n_amb * Am[:, 0] - Bm[:, 0]) / (n_amb * Am[:, 1] + Bm[:, 0])
        return float(np.mean(np.abs(r) ** 2)) * 100.0

    # ---- Nelder-Mead (vectorized) ----
    def nelder_mead(x0, max_iter=600, tol=1e-9, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        d0 = np.array(x0, dtype=float)
        npar = d0.size
        f0 = objective(d0)
        simplex = [d0.copy()]
        fvals = [f0]
        for i in range(1, npar + 1):
            step = d0.copy()
            step[i - 1] += 0.05 * max(abs(d0[i - 1]), 1.0)
            simplex.append(step)
            fvals.append(objective(step))
        best = 0
        for _ in range(max_iter):
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)
            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    for i in range(1, npar + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            if abs(max(fvals) - min(fvals)) < tol:
                break
        order = np.argsort(fvals)
        return simplex[order[0]].copy(), fvals[order[0]]

    # ---- coordinate descent (vectorized line search) ----
    def coord_descent(x0, max_iter=80, tol=1e-10):
        d = np.array(x0, dtype=float)
        cur = objective(d)
        npar = d.size
        for _ in range(max_iter):
            improved = False
            for i in range(npar):
                base = d.copy()
                base[i] = 0.0
                lo, hi = 0.0, max_layer
                flo = objective(base)
                fhi = objective(d)
                # golden section on [lo,hi]
                gr = (math.sqrt(5.0) - 1.0) / 2.0
                c = hi - gr * (hi - lo)
                dd = base.copy(); dd[i] = c
                fc = objective(dd)
                b = lo + gr * (hi - lo)
                db = base.copy(); db[i] = b
                fb = objective(db)
                for _ in range(40):
                    if fc < fb:
                        hi = b; b = c; fb = fc; c = hi - gr * (hi - lo)
                        dd = base.copy(); dd[i] = c; fc = objective(dd)
                    else:
                        lo = c; c = b; fc = fb; b = lo + gr * (hi - lo)
                        db = base.copy(); db[i] = b; fb = objective(db)
                best_x = c if fc < fb else b
                best_f = min(fc, fb)
                if best_f < cur - tol:
                    d = base.copy(); d[i] = best_x
                    cur = best_f
                    improved = True
            if not improved:
                break
        return d, cur

    # ---- initial guesses ----
    lam_c = 0.5 * (w0 + w1)
    qw = {}
    qw['L'] = lam_c / (4.0 * nL)
    qw['H'] = lam_c / (4.0 * nH)

    starts = []
    # 1: quarter-wave alternating
    d1 = np.array([qw['L'] if m == 'L' else qw['H'] for m in seq])
    starts.append(d1)
    # 2: all low-index quarter
    starts.append(np.array([qw['L']] * n))
    # 3: all high-index quarter
    starts.append(np.array([qw['H']] * n))
    # 4: half-wave low on odd positions
    d4 = np.array([qw['L'] * 2.0 if m == 'L' else qw['H'] for m in seq])
    starts.append(d4)
    # 5: small random
    rng = np.random.default_rng(12345)
    for _ in range(6):
        dr = rng.uniform(0.0, 60.0, size=n)
        starts.append(dr)
    # 6: two-layer-like (first two thick, rest tiny)
    d6 = np.zeros(n); d6[0] = qw['L']; d6[1] = qw['H']
    starts.append(d6)

    # ---- run optimization with deadline guard ----
    best_d = None
    best_f = None
    for d0 in starts:
        if time.time() > DEADLINE - 12.0:
            break
        try:
            dn, fn = nelder_mead(d0)
        except Exception:
            continue
        if best_f is None or fn < best_f:
            best_f = fn; best_d = dn.copy()
        if time.time() > DEADLINE - 12.0:
            break
        try:
            dc, fc = coord_descent(dn)
        except Exception:
            continue
        if fc < best_f:
            best_f = fc; best_d = dc.copy()

    if best_d is None:
        # fallback: single quarter-wave low layer
        return ['L'], [qw['L']]

    # ---- enforce constraints: cap each layer, and trim total ----
    d = np.clip(best_d, 0.0, max_layer)
    # if total exceeds max_total, scale down proportionally (keeps within max_layer)
    total = float(np.sum(d))
    if total > max_total:
        d = d * (max_total / total)
        d = np.clip(d, 0.0, max_layer)

    # ---- drop trailing near-zero layers to keep it clean (optional) ----
    labels = []
    thicks = []
    for i in range(n):
        if d[i] < 1e-6:
            continue
        labels.append(seq[i])
        thicks.append(float(d[i]))

    # ensure at least one layer
    if not labels:
        labels = ['L']
        thicks = [qw['L']]

    # final safety: re-check constraints
    if len(labels) > max_layers:
        labels = labels[:max_layers]
        thicks = thicks[:max_layers]
    while sum(thicks) > max_total and thicks:
        # shrink largest layer
        j = int(np.argmax(thicks))
        thicks[j] *= 0.95
    for i in range(len(thicks)):
        if thicks[i] > max_layer:
            thicks[i] = max_layer

    return labels, thicks
# EVOLVE-BLOCK-END
