# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv


def rudin_problem10() -> np.ndarray:
    k = 10
    n = 2 * k + 1
    t0 = time.time()
    DEADLINE = 600.0

    # ---- Quadrature: 70-pt Gauss-Legendre in cos(theta) x 130 equally spaced phi ----
    # Exact for degree-10 (and the degree-20 products needed for the L4 norm).
    x, w = np.polynomial.legendre.leggauss(70)          # x in (-1,1), weights sum to 2
    phi = (2.0 * np.pi / 130.0) * np.arange(130)
    wphi = 2.0 * np.pi / 130.0

    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    N = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(
        np.sum(np.log(np.arange(k - np.arange(n))) - np.log(np.arange(k + np.arange(n) + 1)))
    ))
    # m = -10..10  -> index i = m + 10
    mvals = np.arange(-k, k + 1)
    am = np.abs(mvals)
    Y0 = N * 1.0
    Ypos = np.sqrt(2.0) * N[am > 0] * lpmv(am[am > 0], k, x) * np.cos(am[am > 0][:, None] * phi[None, :])
    Yneg = np.sqrt(2.0) * N[am < 0] * lpmv(am[am < 0], k, x) * np.sin(am[am < 0][:, None] * phi[None, :])
    # Build basis matrix B[i, j] = Y_{m_i}(x_j, phi_j)
    B = np.empty((n, 70 * 130), dtype=np.float64)
    B[0] = Y0
    B[10 + 1:] = Ypos.ravel()
    B[:10] = Yneg.ravel()
    B = B.T  # B[j, i] = value of basis i at node j

    W = np.repeat(w, 130) * wphi                       # node weights, sum = 4*pi
    Wq = W ** 2                                        # weights for f^2 (f^4)

    def ratio(c):
        f = c @ B                                       # (nodes,)
        l2 = np.sqrt(np.dot(W, f * f))
        l4 = np.sqrt(np.sqrt(np.dot(Wq, f * f * f * f)))
        return l4 / l2 if l2 > 0 else np.inf

    def grad(c):
        f = c @ B
        l2sq = np.dot(W, f * f)
        l4sq = np.dot(Wq, f * f * f * f)
        l4 = np.sqrt(l4sq)
        l2 = np.sqrt(l2sq)
        # d(l4/l2)/dc = (1/(2 l2)) * d(l4sq)/dc - (l4/(2 l2^3)) * d(l2sq)/dc
        g = (2.0 * l4 * (Wq * f * f) @ B) / l2 - (l4 * (2.0 * W * f) @ B) / (l2 ** 3)
        return g

    def project(u):
        nrm = np.linalg.norm(u)
        return u / nrm if nrm > 0 else np.ones(n) / np.sqrt(n)

    def line_search(c, g):
        # projected line search on unit sphere
        best_c, best_f = c, ratio(c)
        step = 0.5
        for _ in range(30):
            cand = project(c - step * g)
            val = ratio(cand)
            if val < best_f:
                best_c, best_f = cand, val
                step *= 1.6
            else:
                step *= 0.5
            if step < 1e-12:
                break
        return best_c, best_f

    def optimize(c0, budget):
        c = project(c0)
        fcur = ratio(c)
        while time.time() < budget:
            g = grad(c)
            gn = np.linalg.norm(g)
            if gn < 1e-10:
                break
            c, fcur = line_search(c, g / gn)
        return c, fcur

    # ---- Diverse initializations ----
    rng = np.random.default_rng(12345)
    inits = []
    inits.append(np.ones(n))
    for m0 in range(n):
        v = np.zeros(n); v[m0] = 1.0
        inits.append(v)
    for _ in range(12):
        inits.append(rng.standard_normal(n))
    # low-m0-weighted
    v = np.zeros(n); v[10] = 0.1; v[0] = 1.0; inits.append(v)
    v = np.zeros(n); v[10] = 0.1; v[11] = 1.0; inits.append(v)
    # high-m emphasis
    v = np.zeros(n); v[0] = 1.0; v[1] = 1.0; inits.append(v)
    v = np.zeros(n); v[19] = 1.0; v[20] = 1.0; inits.append(v)

    best_c, best_f = None, np.inf
    n_init = len(inits)
    # per-start budget with margin, but never past DEADLINE
    per = (DEADLINE - (time.time() - t0) - 20.0) / max(1, n_init)
    for v in inits:
        if time.time() > DEADLINE - 30.0:
            break
        c, f = optimize(v, time.time() + per)
        if f < best_f:
            best_f, best_c = f, c
        per = (DEADLINE - (time.time() - t0) - 20.0) / max(1, n_init)  # recompute

    if best_c is None:
        return np.ones(n, dtype=np.float64)

    # polish best a bit more if time remains
    if time.time() < DEADLINE - 30.0:
        c, f = optimize(best_c, DEADLINE - 30.0)
        if f < best_f:
            best_f, best_c = f, c

    out = np.asarray(best_c, dtype=np.float64)
    if not (np.isfinite(out).all()) or np.max(np.abs(out)) < 1e-3 or np.max(np.abs(out)) > 1e5:
        out = np.ones(n, dtype=np.float64)
    return out


# EVOLVE-BLOCK-END
