# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d


def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s


def _entropy_grad(q):
    m = q > 1e-300
    H = -np.sum(q[m] * np.log2(q[m]))
    dq = np.zeros_like(q)
    dq[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return H, dq


def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    D = _entropy_grad(qc)[0]
    M = max(_entropy_grad(qa)[0], _entropy_grad(qb)[0], _entropy_grad(qd)[0])
    if M < 1e-12:
        return -1e9
    return D / M


def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1

    def f_and_grad(z):
        P = _softmax(z)
        qa = np.bincount(idx_a, weights=P, minlength=la)
        qb = np.bincount(idx_b, weights=P, minlength=lb)
        qc = np.bincount(idx_c, weights=P, minlength=lc)
        qd = np.bincount(idx_d, weights=P, minlength=ld)
        Ha, ga = _entropy_grad(qa)
        Hb, gb = _entropy_grad(qb)
        Hc, gc = _entropy_grad(qc)
        Hd, gd = _entropy_grad(qd)
        D = Hc
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return -1e9, np.zeros_like(z)
        # subgradient: pick the projection achieving M
        if Ha >= Hb and Ha >= Hd:
            gM = ga
        elif Hb >= Hd:
            gM = gb
        else:
            gM = gd
        df_dP = gc / M - (D / (M * M)) * gM
        mean = np.dot(df_dP, P)
        grad_z = P * (df_dP - mean)
        return D / M, grad_z

    def fun(z):
        return -f_and_grad(z)[0]

    def fprime(z):
        return -f_and_grad(z)[1]

    return fun, fprime


def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy_grad(qa)[0], _entropy_grad(qb)[0], _entropy_grad(qd)[0])
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    fun, fprime = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d)

    start_time = time.time()
    deadline = start_time + 620.0  # safety margin under the 630s kill

    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    starts = []

    # 1. Uniform
    starts.append(np.zeros(100))

    # 2. Fixed-x and fixed-y lines
    for x0 in range(n):
        P = np.zeros((n, n))
        P[:, x0] = 1.0 / n
        starts.append(np.log(P.ravel() + 1e-12))
    for y0 in range(n):
        P = np.zeros((n, n))
        P[y0, :] = 1.0 / n
        starts.append(np.log(P.ravel() + 1e-12))

    # 3. Diagonal lines x - y = c
    for c in range(-9, 10):
        P = np.zeros((n, n))
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P = P / P.sum()
            starts.append(np.log(P.ravel() + 1e-12))

    # 4. Diagonal lines x + y = c
    for c in range(0, 19):
        P = np.zeros((n, n))
        for y in range(n):
            x = c - y
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P = P / P.sum()
            starts.append(np.log(P.ravel() + 1e-12))

    # 5. Anisotropic Gaussians - expanded grid of centers and scales
    # Focus on centers that might break symmetry
    centers = [1, 2, 3, 4, 5, 6, 7, 8]
    scales = [0.5, 0.8, 1.2, 1.8, 2.5, 3.5]
    for cx in centers:
        for cy in centers:
            for sx in scales:
                for sy in scales:
                    logP = -(((X - cx) ** 2) / (sx * sx) + ((Y - cy) ** 2) / (sy * sy))
                    z = logP.ravel()
                    z = z - z.max()
                    starts.append(z)

    # 6. Block distributions (concentrating mass on sub-grids)
    # This helps explore regions where mass is concentrated in corners or edges
    block_sizes = [2, 3, 4, 5]
    for bs in block_sizes:
        for by in range(0, n - bs + 1):
            for bx in range(0, n - bs + 1):
                P = np.zeros((n, n))
                P[by:by+bs, bx:bx+bs] = 1.0
                P = P / P.sum()
                starts.append(np.log(P.ravel() + 1e-12))

    # 7. Random starts
    rng = np.random.default_rng(42)
    for s in (0.5, 1.0, 2.0, 3.0, 4.0):
        for _ in range(15):
            starts.append(rng.normal(0.0, s, 100))

    # Evaluate raw value at a start (no grad)
    def raw_at(z):
        P = _softmax(z)
        return _value_from_P(P, idx_a, idx_b, idx_c, idx_d)

    best_f = -1e9
    best_z = np.zeros(100)

    # Seed best with evaluated starts (cheap floor)
    # Evaluate a subset of starts first to find a good baseline quickly
    for z in starts[:30]:
        f = raw_at(z)
        if f > best_f:
            best_f = f
            best_z = z.copy()

    # Optimize each start within the time budget
    # We have ~100-200 starts. Each L-BFGS-B run is fast (<1s usually).
    # 600s budget allows for many runs.
    for z0 in starts:
        if time.time() > deadline - 10.0:
            break
        z0 = z0.copy()
        try:
            res = minimize(fun, z0, jac=fprime, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 2000,
                                    'ftol': 1e-10, 'gtol': 1e-8})
            zf = res.x
            f = -res.fun
            if np.isfinite(f) and f > best_f:
                best_f = f
                best_z = zf.copy()
        except Exception:
            f = raw_at(z0)
            if f > best_f:
                best_f = f
                best_z = z0.copy()

    # Final polish if time remains
    if time.time() < deadline - 15.0:
        try:
            res = minimize(fun, best_z, jac=fprime, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 5000,
                                    'ftol': 1e-12, 'gtol': 1e-10})
            f = -res.fun
            if np.isfinite(f) and f > best_f:
                best_f = f
                best_z = res.x.copy()
        except Exception:
            pass

    # One more polish round if still time
    if time.time() < deadline - 20.0:
        try:
            res = minimize(fun, best_z, jac=fprime, method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 3000,
                                    'ftol': 1e-13, 'gtol': 1e-11})
            f = -res.fun
            if np.isfinite(f) and f > best_f:
                best_f = f
                best_z = res.x.copy()
        except Exception:
            pass

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P


# EVOLVE-BLOCK-END
