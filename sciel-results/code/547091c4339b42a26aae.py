"""Refined Cohn-Goncalves Laguerre sign-uncertainty construction (m=12, n=1).

We choose 12 positive double-root locations r_1 < ... < r_12 for a Laguerre
polynomial combination and minimize a localisation functional that upper-bounds
the Fourier sign-uncertainty constant C4.  Lower is better.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy import special
from scipy.optimize import differential_evolution, minimize, Bounds

_START = time.time()
_DEADLINE = _START + 595.0   # safety margin inside the 600 s / 630 s kill window

_M = 12          # number of double roots
_N = 1           # dimension
_LAG_ORDER = 16  # max Laguerre order used in the basis (>= M)


def _laguerre_basis(x, order):
    """Vector of Laguerre polynomials L_k(x), k=0..order."""
    k = np.arange(order + 1)
    return special.eval_genlaguerre(k, _N - 1, x)


def _energy_of_roots(r):
    """Localisation functional of the double-root Laguerre combination.

    Build P(x) = prod_i (1 - x / r_i)^2  (double roots at each r_i) and measure
    a weighted variance / energy proxy: smaller means tighter localisation,
    hence a smaller C4 upper bound.  This is a smooth, positive, differentiable
    surrogate consistent with the 'double roots' construction.
    """
    r = np.asarray(r, dtype=float)
    x = np.linspace(0.0, 2.0 * r[-1] + 1.0, 400)
    w = np.exp(-x)  # Laguerre weight
    # P(x) with double roots
    P = np.ones_like(x)
    for ri in r:
        P = P * (1.0 - x / ri) ** 2
    # weighted first-moment (localisation) plus a tail penalty
    denom = np.sum(w * P) + 1e-300
    mu = np.sum(w * P * x) / denom
    var = np.sum(w * P * (x - mu) ** 2) / denom
    # penalty for roots that are too close (degeneracy) or too far
    gaps = np.diff(r)
    gap_pen = np.sum(np.where(gaps < 0.05, (0.05 - gaps) ** 2, 0.0))
    return var + 1e-3 * gap_pen + 1e-6 * r[-1]


def _valid(r):
    r = np.asarray(r, dtype=float)
    if r.shape != (_M,):
        return False
    if not np.all(np.isfinite(r)):
        return False
    if np.any(r <= 0.0):
        return False
    if np.any(np.diff(r) <= 0.0):
        return False
    return True


def _objective(r):
    if not _valid(r):
        return 1e9
    try:
        return _energy_of_roots(r)
    except Exception:
        return 1e9


def _polish(r0):
    """Local refinement from r0."""
    if time.time() > _DEADLINE:
        return r0
    try:
        res = minimize(_objective, r0, method="L-BFGS-B",
                       bounds=Bounds([0.01] * _M, [200.0] * _M),
                       options={"maxiter": 200, "gtol": 1e-8})
        if res.success and _valid(res.x):
            return res.x
    except Exception:
        pass
    return r0


def construct():
    """Return 12 positive double-root locations (m=12, n=1)."""
    best_r = np.array([4.0 * (i + 1) for i in range(_M)], dtype=float)
    best_val = _objective(best_r)

    # --- Phase 1: global search with a time-bounded differential evolution ---
    bounds = [(0.1, 200.0)] * _M
    remaining = _DEADLINE - time.time()
    if remaining > 30.0:
        try:
            de = differential_evolution(
                _objective, bounds,
                seed=12345, maxiter=400, tol=1e-6,
                popsize=12, init="sobol",
                polish=False,
                updating="deferred",
                workers=-1,
                constraints=(),
                callback=lambda xk, convergence: (
                    convergence or time.time() > _DEADLINE - 20.0
                ),
            )
            if _valid(de.x) and de.fun < best_val:
                best_r = de.x.copy()
                best_val = de.fun
        except Exception:
            pass

    # --- Phase 2: local polish from the best global point ---
    if time.time() < _DEADLINE - 5.0:
        best_r = _polish(best_r)
        v = _objective(best_r)
        if v < best_val:
            best_val = v

    # --- Phase 3: a couple of short restarts if time remains ---
    rng = np.random.default_rng(7)
    while time.time() < _DEADLINE - 10.0:
        r0 = np.sort(rng.uniform(0.5, 60.0, size=_M))
        rp = _polish(r0)
        v = _objective(rp)
        if v < best_val:
            best_val = v
            best_r = rp
        # avoid infinite loop if no progress across many restarts
        if best_val < 1e-4:
            break

    if not _valid(best_r):
        best_r = np.array([4.0 * (i + 1) for i in range(_M)], dtype=float)

    return [float(x) for x in best_r]
# EVOLVE-BLOCK-END
