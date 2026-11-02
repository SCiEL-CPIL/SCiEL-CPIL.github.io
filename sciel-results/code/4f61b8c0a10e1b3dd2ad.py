"""Uncertainty inequality: tune Hermite-Gaussian to minimize the C4 upper bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_BUDGET = 210.0
_MARGIN = 25.0

# --- Hermite (physicists) polynomials up to degree 12 (exact integer coeffs) ---
_H = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, -3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 24, 0, -12, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 120, 0, -60, 0, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 720, 0, -360, 0, 30, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5040, 0, -2520, 0, 35, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 40320, 0, -20160, 0, 280],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 362880, 0, -181440, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3628800, 0, -1814400],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39916800, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 479001600],
]
_H = np.array(_H, dtype=float)


def _a12(a0, a4, a8):
    # g(0)=0  ->  a0 - 3 a4 - 105 a8 - 945 a12 = 0
    return -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0


def _coeffs(c):
    a0, a4, a8 = c
    return np.array([a0, a4, a8, _a12(a0, a4, a8)], dtype=float)


def _poly_moments(coeffs, N):
    # returns m[k] = <g^2 x^k> for k=0..N  (exact via Hermite moments)
    m = np.zeros(N + 1)
    for i in range(4):
        ci = coeffs[i]
        if ci == 0.0:
            continue
        for j in range(4):
            cj = coeffs[j]
            if cj == 0.0:
                continue
            w = ci * cj
            d = i + j
            if d > N:
                continue
            for k in range(d, N + 1):
                m[k] += w * _H[k, d]
    return m


def _bound(c):
    coeffs = _coeffs(c)
    N = 24
    m = _poly_moments(coeffs, N)
    A = m[0]
    if A <= 0.0:
        return np.inf
    # A(|g|^4) = <g^4>  (x^0 moment of g^4)
    A4 = m[0]
    if A4 <= 0.0:
        return np.inf
    # A(|g|^4)(x) = <g^4 (x-y)^2> = 2x^2 m[0] - 2x m[1] + m[2]
    b = -2.0 * m[1]
    cc = m[2]
    disc = b * b - 4.0 * 2.0 * A4 * cc
    if disc > 0.0:
        r1 = (-b - np.sqrt(disc)) / (4.0 * A4)
        r2 = (-b + np.sqrt(disc)) / (4.0 * A4)
        roots = [r1, r2]
    else:
        roots = []
    # roots of x^2 A(|g|^4)(x)  (A4 x^2 + b x + cc)
    disc2 = b * b - 4.0 * A4 * cc
    if disc2 > 0.0:
        s1 = (-b - np.sqrt(disc2)) / (2.0 * A4)
        s2 = (-b + np.sqrt(disc2)) / (2.0 * A4)
        roots += [s1, s2]
    else:
        roots += [0.0]
    xgrid = np.array(roots, dtype=float)
    A4x = 2.0 * A4 * xgrid * xgrid + b * xgrid + cc
    valid = A4x > 1e-15
    if not np.any(valid):
        return np.inf
    xg = xgrid[valid]
    denom = np.sqrt(A4x[valid])
    frac = A * A / denom
    return float(np.max(frac))


def _objective(c):
    b = _bound(c)
    return b if np.isfinite(b) else 1e12


def construct():
    deadline = _START + _BUDGET - _MARGIN
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_c = seed.copy()
    best_v = _bound(seed)

    def consider(c):
        nonlocal best_c, best_v
        v = _bound(c)
        if np.isfinite(v) and v < best_v:
            best_v = v
            best_c = c.copy()

    consider(seed)

    # --- global grid over a box around the seed ---
    if time.time() < deadline:
        lo = np.array([-1.5, -0.02, -0.02])
        hi = np.array([1.5, 0.02, 0.02])
        rng = np.random.default_rng(0)
        # coarse structured grid
        for n in (14, 10, 8):
            if time.time() > deadline:
                break
            g0 = np.linspace(lo[0], hi[0], n)
            g1 = np.linspace(lo[1], hi[1], n)
            g2 = np.linspace(lo[2], hi[2], n)
            for x0 in g0:
                for x1 in g1:
                    for x2 in g2:
                        consider(np.array([x0, x1, x2]))
        # random jitter
        for _ in range(6000):
            if time.time() > deadline:
                break
            c = lo + (hi - lo) * rng.random(3)
            consider(c)

    # --- local refinement from best + a set of good starts ---
    starts = [best_c.copy()]
    rng2 = np.random.default_rng(1)
    for _ in range(24):
        starts.append(lo + (hi - lo) * rng2.random(3))

    if time.time() < deadline:
        for s in starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(_objective, s, method="L-BFGS-B",
                               options={"maxiter": 400, "maxls": 100,
                                        "ftol": 1e-12, "gtol": 1e-10})
                if res.success or res.fun < best_v:
                    consider(res.x)
            except Exception:
                pass

    # --- high-precision polish (tighter tolerances, more iters) ---
    if time.time() < deadline:
        try:
            res = minimize(_objective, best_c, method="L-BFGS-B",
                           options={"maxiter": 2000, "maxls": 200,
                                    "ftol": 1e-15, "gtol": 1e-13})
            consider(res.x)
        except Exception:
            pass

    # --- final local grid polish around the best point ---
    if time.time() < deadline:
        eps = 1e-3
        center = best_c.copy()
        for n in (7,):
            for dx0 in np.linspace(-eps, eps, n):
                for dx1 in np.linspace(-eps, eps, n):
                    for dx2 in np.linspace(-eps, eps, n):
                        consider(center + np.array([dx0, dx1, dx2]))

    return best_c
# EVOLVE-BLOCK-END
