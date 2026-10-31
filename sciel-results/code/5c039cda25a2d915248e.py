"""Uncertainty inequality: analytic-gradient multi-start optimization.

Test function g(x) = exp(-x^2/2) * (a0 H0 + a4 H4 + a8 H8 + a12 H12)
with g(0)=0 fixing a12 = (a0 - 3 a4 + 105 a8)/10395.

A(g) = int |g(x)| dx, and by Fourier invariance A(F(g)) = A(g), so the
upper bound on C4 is A(g)^2.  We minimize this over (a0, a4, a8).

Strategy: analytic gradient of A(g) via sign-based rule, then
  1) coarse global grid to bracket the basin,
  2) multi-start L-BFGS-B from grid + random + best-known points,
  3) final polish.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

_T0 = time.time()
_DEADLINE = 190.0  # safety margin before 210 s kill

def _time_left():
    return _DEADLINE - (time.time() - _T0)

# Precompute quadrature nodes (Gauss-Hermite, 150 points)
_N = 150
_U, _W = np.polynomial.hermite.hermgauss(_N)
_X = np.sqrt(2.0) * _U          # x = sqrt(2) u
_H0 = np.ones_like(_X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_SQRT2 = np.sqrt(2.0)

# a12 = (a0 - 3 a4 + 105 a8) / 10395
_C12 = 1.0 / 10395.0
# Contribution of a12*H12 to g:
#   d(a12)/da0 * H12 = _C12 * H12
#   d(a12)/da4 * H12 = -3*_C12 * H12
#   d(a12)/da8 * H12 = 105*_C12 * H12
_D12_H12 = _C12 * _H12
_D12_4_H12 = -3.0 * _C12 * _H12
_D12_8_H12 = 105.0 * _C12 * _H12

def _A_and_grad(coeffs):
    """Return (A, grad_A) where A = sqrt(2) * sum w_i |g(x_i)|."""
    a0, a4, a8 = coeffs
    # g(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    a12 = (a0 - 3.0*a4 + 105.0*a8) * _C12
    g = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12

    A = _SQRT2 * np.dot(_W, np.abs(g))

    # grad A / da_j = sqrt(2) * sum w_i * sign(g_i) * dg/d(a_j)_i
    sign_g = np.sign(g)
    # dg/da0 = H0 + C12*H12
    # dg/da4 = H4 - 3*C12*H12
    # dg/da8 = H8 + 105*C12*H12
    g0 = _H0 + _D12_H12
    g4 = _H4 + _D12_4_H12
    g8 = _H8 + _D12_8_H12

    wg_sign = _W * sign_g
    grad0 = _SQRT2 * np.dot(wg_sign, g0)
    grad4 = _SQRT2 * np.dot(wg_sign, g4)
    grad8 = _SQRT2 * np.dot(wg_sign, g8)

    return A, np.array([grad0, grad4, grad8])

def _objective(coeffs):
    """Objective = A^2 (the upper bound on C4)."""
    A, _ = _A_and_grad(coeffs)
    return A * A

def _objective_grad(coeffs):
    A, gA = _A_and_grad(coeffs)
    return A * A, 2.0 * A * gA

def construct():
    best_x = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _objective(best_x)

    def _try_point(x0):
        nonlocal best_x, best_val
        if _time_left() < 2.0:
            return
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, options={'maxiter': 2000,
                                               'ftol': 1e-16,
                                               'gtol': 1e-14,
                                               'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Phase 1: coarse global grid ---
    # The known best is around (-1.13, 0.04, 0.0003).  Search a wider box.
    a0_range = np.linspace(-5.0, 5.0, 11)
    a4_range = np.linspace(-2.0, 2.0, 9)
    a8_range = np.linspace(-0.01, 0.01, 7)

    grid_starts = []
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                grid_starts.append([a0, a4, a8])

    # Sort grid by objective to visit promising points first
    grid_starts.sort(key=lambda p: _objective(p))

    for pt in grid_starts:
        if _time_left() < 5.0:
            break
        _try_point(np.array(pt))

    # --- Phase 2: random starts in a tighter box around the best ---
    rng = np.random.default_rng(12345)
    for i in range(80):
        if _time_left() < 5.0:
            break
        # Mix: some near best, some in wider box
        if i % 3 == 0:
            start = best_x + rng.normal(0, 0.1, 3)
        elif i % 3 == 1:
            start = rng.uniform([-3, -1, -0.005], [3, 1, 0.005])
        else:
            start = rng.uniform([-8, -3, -0.02], [8, 3, 0.02])
        _try_point(start)

    # --- Phase 3: final polish from best ---
    if _time_left() > 5.0:
        try:
            res = minimize(_objective_grad, best_x, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000,
                                               'ftol': 1e-18,
                                               'gtol': 1e-16,
                                               'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Phase 4: one more Nelder-Mead polish (handles non-smoothness at sign changes) ---
    if _time_left() > 5.0:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14,
                                    'fatol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return np.array(best_x)
