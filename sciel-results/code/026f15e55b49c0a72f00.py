"""Optimize Hermite coefficients to reduce the uncertainty constant C4."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

_DEADLINE = 205.0  # safety margin before the 210 s kill
_START = time.time()


def _hermite_poly(n):
    """Probabilists' Hermite polynomial He_n(x) (physicist H_n)."""
    x = np.arange(n + 1)
    return eval_hermite(n, x)


# Precompute the four probabilists' Hermite polynomials on a grid
_N = 4096
_XMIN = -6.0
_XMAX = 6.0
_X = np.linspace(_XMIN, _XMAX, _N)
_DX = _X[1] - _X[0]
_GAUSS = np.exp(-0.5 * _X * _X)
_POLY = [_hermite_poly(n) for n in (0, 4, 8, 12)]
_POLY0 = _POLY[0] * _GAUSS
_POLY4 = _POLY[1] * _GAUSS
_POLY8 = _POLY[2] * _GAUSS
_POLY12 = _POLY[3] * _GAUSS
# a12 multiplier so that g(0)=0:  a0 + a4*He4(0) + a8*He8(0) + a12*He12(0) = 0
_H0_0 = float(_POLY[0][0])
_H4_0 = float(_POLY[1][4])
_H8_0 = float(_POLY[2][8])
_H12_0 = float(_POLY[3][12])


def _g_values(c):
    """Return g(x) on the grid for free coefficients c = (a0,a4,a8)."""
    a0, a4, a8 = c
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    return (a0 * _POLY0 + a4 * _POLY4 + a8 * _POLY8 + a12 * _POLY12)


def _objective(c):
    """Smooth surrogate of C4 = ||g||_1 / min|g| via log-sum-exp + epigraph t."""
    g = _g_values(c)
    norm1 = np.sum(np.abs(g)) * _DX
    # log-sum-exp of |g| for a smooth max
    m = np.max(np.abs(g))
    if m < 1e-300:
        return 1e30
    lse = m + math.log(np.sum(np.exp(np.abs(g) - m)))
    # log C4 = log(norm1) - log(min|g|) ; use soft-min via -lse of -|g|
    # min|g| approx = exp(-lse(-|g|)) = exp(-(m2 + log sum exp(-|g|-m2)))
    m2 = np.max(-g)  # = -min g
    # soft-min of |g|:  sm = exp(-( m3 + logsumexp(-|g| - m3) ))
    m3 = np.max(-np.abs(g))
    lse_neg = m3 + math.log(np.sum(np.exp(-np.abs(g) - m3)))
    soft_min = math.exp(-lse_neg)
    if soft_min < 1e-300:
        return 1e30
    return norm1 / soft_min


def _objective_exact(c):
    """True C4 (nonsmooth) for final polish comparison."""
    g = _g_values(c)
    norm1 = np.sum(np.abs(g)) * _DX
    mn = np.min(np.abs(g))
    if mn < 1e-300:
        return 1e30
    return norm1 / mn


def _polish(c, tol=1e-9, max_iter=4000):
    """Nelder-Mead style local refinement with adaptive step."""
    from scipy.optimize import minimize
    try:
        res = minimize(_objective, c, method='Nelder-Mead',
                       options={'xatol': tol, 'fatol': tol, 'maxiter': max_iter,
                                'adaptive': True})
        if res.success or res.fun < _objective(c):
            return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return c


def construct():
    global _START
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_c)
    best_exact = _objective_exact(best_c)

    rng = np.random.default_rng(12345)

    # Multi-start: seed + random + perturbation starts
    starts = [best_c.copy()]
    # deterministic structured starts
    for s in range(40):
        starts.append(rng.normal(0, 1.0, size=3))
    for s in range(20):
        starts.append(best_c + rng.normal(0, 0.3, size=3))
    for s in range(20):
        starts.append(rng.normal(0, 0.5, size=3))

    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            cand = _polish(start, tol=1e-7, max_iter=2000)
            v = _objective(cand)
            if v < best_val:
                best_val = v
                best_c = cand.copy()
        except Exception:
            continue

    # Final high-precision polish on the best candidate
    if time.time() < _DEADLINE - 10:
        try:
            cand = _polish(best_c, tol=1e-10, max_iter=6000)
            if _objective(cand) <= best_val + 1e-9:
                best_c = cand
        except Exception:
            pass

    # Return the three free coefficients (a0, a4, a8)
    return np.array([best_c[0], best_c[1], best_c[2]], dtype=float)
# EVOLVE-BLOCK-END
