"""Combined program: smooth surrogate + multi-start optimization for uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 165.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


# Precompute the polynomial basis on a fixed grid (probabilists' Hermite)
_N = 40001
_X = np.linspace(-12.0, 12.0, _N)
_DX = _X[1] - _X[0]
_X2 = _X * _X
_EXP = np.exp(-_X2 / 2.0)
# Probabilists' Hermite: He0=1, He4=x^4-6x^2+3, He8=x^8-28x^6+210x^4-420x^2+105,
# He12=x^12-42x^10+1260x^8-16800x^6+113400x^4-302400x^2+10395
_H0 = np.ones_like(_X)
_H4 = _X2 * _X2 - 6.0 * _X2 + 3.0
_H8 = _X2 * _X2 * _X2 * _X2 - 28.0 * _X2 * _X2 * _X2 + 210.0 * _X2 * _X2 - 420.0 * _X2 + 105.0
_H12 = (_X2 * _X2 * _X2 * _X2 * _X2 * _X2
        - 42.0 * _X2 * _X2 * _X2 * _X2 * _X2
        + 1260.0 * _X2 * _X2 * _X2 * _X2
        - 16800.0 * _X2 * _X2 * _X2
        + 113400.0 * _X2 * _X2
        - 302400.0 * _X2
        + 10395.0)

# Hermite values at 0 for the g(0)=0 constraint (probabilists')
_H0_0 = 1.0
_H4_0 = -3.0
_H8_0 = 105.0
_H12_0 = -10395.0


def _c12(c0, c4, c8):
    return -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0


def _f_values(c0, c4, c8):
    """Return the function values f(x) = p(x) exp(-x^2/2) on the fixed grid."""
    c12 = _c12(c0, c4, c8)
    p = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    return p * _EXP


def _bound(c0, c4, c8):
    """
    Upper bound on A(f)*A(F(f)).
    For degrees 0,4,8,12 the FT phase is (+1), so |F(f)(xi)| = |f(xi)|
    (self-reciprocal up to the same normalization). The product is therefore
    proportional to ||f||_1^2. We compute ||f||_1 on the fixed grid.
    """
    f = _f_values(c0, c4, c8)
    A_f = np.sum(np.abs(f)) * _DX
    # By the self-reciprocal symmetry A_F == A_f (same grid, same shape).
    # The constant factor is identical for the seed and any candidate, so
    # minimizing A_f * A_f == A_f^2 is equivalent. Return A_f^2.
    return A_f * A_f


def _bound_smooth(c0, c4, c8, tau=1e-3):
    """
    Smooth surrogate for ||f||_1 using soft-abs: |z| ~ sqrt(z^2 + tau^2).
    This removes the kink at f=0, giving a smooth objective for the optimizer.
    """
    f = _f_values(c0, c4, c8)
    soft_abs = np.sqrt(f * f + tau * tau)
    A_f = np.sum(soft_abs) * _DX
    return A_f * A_f


def _refine(params, maxiter=3000, xatol=1e-12, fatol=1e-14):
    """Local Nelder-Mead refinement with the smooth surrogate, then exact check."""
    try:
        res = minimize(
            lambda v: _bound_smooth(v[0], v[1], v[2]),
            params,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                     'adaptive': True, 'initial_simplex': None}
        )
        return res.x
    except Exception:
        return params


def construct():
    """
    Find the three free coefficients (c0, c4, c8) minimizing the upper bound.
    Strategy: smooth-surrogate local optimization from the seed, a few random
    restarts, and a final high-precision polish.
    """
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_params = seed.copy()
    best_val = _bound(*seed)

    # 1) Local refinement from the seed using the smooth surrogate
    if _time_left() > 40:
        cand = _refine(seed, maxiter=4000, xatol=1e-13, fatol=1e-15)
        val = _bound(*cand)
        if val < best_val:
            best_val = val
            best_params = cand

    # 2) A few random restarts with local refinement
    if _time_left() > 50:
        rng = np.random.default_rng(12345)
        for i in range(8):
            if _time_left() < 25:
                break
            # Perturb around the current best, shrinking scale over iterations
            scale = 0.5 / (i + 1)
            perturb = best_params + rng.normal(0.0, scale, size=3)
            cand = _refine(perturb, maxiter=2000, xatol=1e-12, fatol=1e-14)
            val = _bound(*cand)
            if val < best_val:
                best_val = val
                best_params = cand

    # 3) Final high-precision polish with tighter tolerances
    if _time_left() > 20:
        cand = _refine(best_params, maxiter=6000, xatol=1e-14, fatol=1e-16)
        val = _bound(*cand)
        if val < best_val:
            best_val = val
            best_params = cand

    # 4) One more ultra-tight polish if time remains
    if _time_left() > 12:
        cand = _refine(best_params, maxiter=4000, xatol=1e-15, fatol=1e-17)
        val = _bound(*cand)
        if val < best_val:
            best_val = val
            best_params = cand

    return np.array(best_params)
# EVOLVE-BLOCK-END
