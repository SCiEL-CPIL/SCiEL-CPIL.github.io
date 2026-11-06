"""Growth-index compact ansatz: fit 6 coefficients to a GR reference.

Design (kept within the constant-γ / compact-ansatz family):
  * A fast, self-contained general-relativistic reference computes f and
    the normalized growth factor D(a)/D(1) for smooth CPL dark-energy models.
  * The six coefficients are fit by minimizing a SMOOTH surrogate of the
    min/max objective: the sum of relative errors of f and D over a grid of
    validation cosmologies and scale factors.  A smooth sum is far more
    tractable than a direct max and still drives the worst point down.
  * Differential Evolution gives a robust global fit; Nelder-Mead polishes
    the smooth surrogate.  A deadline guard returns the best feasible result.
"""

# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 78.0          # seconds wall-clock budget (safety margin vs 90s kill)
_BIG = 1e12


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_models():
    # (Om0, w0, wa)  -- a spread of smooth CPL dark-energy cosmologies.
    return [
        (0.30, -1.0, 0.0),
        (0.28, -1.0, 0.0),
        (0.315, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.5),
        (0.30, -0.9, 0.5),
        (0.30, -1.1, 0.5),
        (0.25, -0.95, 0.3),
        (0.35, -1.0, 0.4),
        (0.30, -0.85, 0.8),
        (0.30, -1.2, -0.3),
        (0.30, -1.0, 1.0),
        (0.28, -0.9, 0.6),
        (0.32, -1.05, -0.4),
        (0.30, -0.95, 0.0),
        (0.30, -1.0, 0.2),
        (0.30, -1.0, -0.2),
        (0.30, -1.1, 0.3),
        (0.30, -0.9, -0.3),
        (0.30, -1.0, 0.7),
    ]


def _build_grid():
    # log-spaced scale factors; a=1 is exact (D=1, f=1) and skipped.
    a = np.unique(np.round(np.geomspace(0.08, 1.0, 200), 8))
    a = a[a < 0.9999]
    return a


def _compute_ref(Om0, w0, wa, a):
    """GR reference: growth rate f and normalized D(a)/D(1) at scale factors a."""
    a = np.asarray(a, dtype=float)
    n = a.size
    lnA = np.log(a)
    # matter
    Om_a = Om0 * a ** 3
    q = 1.0 - Om_a
    # dark-energy scale factor
    if abs(wa) < 1e-12:
        de_a = a ** (-3.0 * (1.0 + w0))
    else:
        de_a = np.exp(3.0 * wa * (1.0 - a)) * a ** (-3.0 * (1.0 + w0))
    de = 1.0 - Om0 - de_a          # de(1)=0
    H2 = Om_a + de
    H2d = 3.0 * Om_a + 3.0 * (1.0 + w0 + wa * (1.0 - a)) * de
    # matter-only (EdS) growth
    f0 = a ** 1.5
    D0 = a ** 1.5
    # GR second-order ODE for D
    y = np.zeros((2, n))
    y[0, 0] = 1.0
    y[1, 0] = 1.5
    for i in range(n - 1):
        if H2[i] <= 1e-12 or H2[i + 1] <= 1e-12:
            break
        d = lnA[i + 1] - lnA[i]
        D0i, f0i = D0[i], f0[i]
        A1 = (3.0 - H2d[i] / H2[i]) / 2.0
        B1 = (3.0 - H2d[i + 1] / H2[i + 1]) / 2.0
        C1 = 3.0 / 2.0 * (1.0 - H2[i + 1] / H2[i])
        D1 = 3.0 * (1.0 - f0i * f0[i]) / 2.0
        A2 = (3.0 + H2d[i] / H2[i]) / 2.0
        B2 = (3.0 + H2d[i + 1] / H2[i + 1]) / 2.0
        C2 = 3.0 / 2.0 * (1.0 - H2[i] / H2[i + 1])
        D2 = 3.0 * (1.0 - f0[i] * f0i) / 2.0
        D = y[0, i]; F = y[1, i]
        y[0, i + 1] = D + 0.5 * d * (D1 + (A1 * D + B1 * F))
        y[1, i + 1] = F + 0.5 * d * (D2 + (A2 * D + B2 * F))
    D = y[0]
    D = D / D[-1]                    # normalize D(1)=1
    f = D / f0                       # f = Om_a * D'
    return f, D


def _compute_app(c, Om0, w0, wa, a):
    """Ansatz: gamma, f=Om^gamma, and D via exp(-integral f)."""
    a = np.asarray(a, dtype=float)
    lnA = np.log(a)
    Om_a = Om0 * a ** 3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = (c[0] + c[1] * q + c[2] * q * q + c[3] * u +
             c[4] * wa * a + c[5] * u * q)
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return None, None
    f = Om_a ** gamma
    # normalized growth factor: D/D(1)=exp(-int_0^ln a f dx)
    S = np.empty(a.size)
    S[0] = 0.0
    S[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lnA))
    D = np.exp(-S)
    return f, D


def _surrogate(c, models, grid):
    """Smooth surrogate: sum of relative errors of f and D over the grid."""
    total = 0.0
    for (Om0, w0, wa) in models:
        f_r, D_r = _compute_ref(Om0, w0, wa, grid)
        f_a, D_a = _compute_app(c, Om0, w0, wa, grid)
        if f_a is None:
            return _BIG
        total += float(np.sum(np.abs(f_a / f_r - 1.0)))
        total += float(np.sum(np.abs(D_a / D_r - 1.0)))
    return total


def _seed_value(c):
    """Worst relative error over the grid (for reporting / sanity)."""
    worst = 0.0
    for (Om0, w0, wa) in _build_models():
        f_r, D_r = _compute_ref(Om0, w0, wa, _build_grid())
        f_a, D_a = _compute_app(c, Om0, w0, wa, _build_grid())
        if f_a is None:
            return _BIG
        worst = max(worst,
                    float(np.max(np.abs(f_a / f_r - 1.0))),
                    float(np.max(np.abs(D_a / D_r - 1.0))))
    return worst


def construct():
    global _MODELS, _GRID, _REF_CACHE
    if not hasattr(construct, "_MODELS"):
        construct._MODELS = _build_models()
        construct._GRID = _build_grid()
    models = construct._MODELS
    grid = construct._GRID

    def obj(c):
        if _remaining() < 2.0:
            return _BIG
        return _surrogate(c, models, grid)

    # seed (constant-gamma) and its score
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    seed_score = _seed_value(c0)
    best_c = c0.copy()
    best_obj = obj(c0)

    # global search
    bounds = [(-1.0, 1.0)] * 6
    if _remaining() > 18.0:
        try:
            res = differential_evolution(
                obj, bounds, seed=20240517, tol=1e-6, maxiter=40,
                popsize=14, mutation=(0.4, 1.0), recombination=0.8,
                polish=False, init='sobol', updating='immediate',
                workers=-1, seed=20240517)
            if res is not None and np.all(np.isfinite(res.x)) and res.fun < best_obj:
                best_c = np.asarray(res.x, dtype=float)
                best_obj = float(res.fun)
        except Exception:
            pass

    # local polish on the smooth surrogate
    if _remaining() > 6.0:
        try:
            r2 = minimize(obj, best_c, method='Nelder-Mead',
                          options={'maxiter': 4000, 'xatol': 1e-8,
                                   'fatol': 1e-9, 'adaptive': True})
            if r2 is not None and np.all(np.isfinite(r2.x)) and r2.fun < best_obj:
                best_c = np.asarray(r2.x, dtype=float)
                best_obj = float(r2.fun)
        except Exception:
            pass

    # final polish with tighter tolerances
    if _remaining() > 3.0:
        try:
            r3 = minimize(obj, best_c, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10,
                                   'fatol': 1e-11, 'adaptive': True})
            if r3 is not None and np.all(np.isfinite(r3.x)) and r3.fun < best_obj:
                best_c = np.asarray(r3.x, dtype=float)
                best_obj = float(r3.fun)
        except Exception:
            pass

    # enforce bounds and feasibility; fall back to seed if needed
    best_c = np.clip(best_c, -10.0, 10.0)
    if not np.all(np.isfinite(best_c)) or _seed_value(best_c) >= _BIG:
        best_c = c0

    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
