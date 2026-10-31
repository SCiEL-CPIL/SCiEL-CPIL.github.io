"""Exact analytic Hermite-Gaussian uncertainty bound with analytic gradient.

The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}, so every
integral in the bound reduces to sums of 2^n n! via Hermite orthogonality.  This
gives a smooth closed-form objective with an exact gradient (no FFT, no quadrature,
no finite differences), enabling a dense multi-start search plus a tight polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 8.0

# --- Exact Hermite-Gaussian Fourier/momentum operators (even parity only) ---
# For p = sum a_n H_n,  g = p e^{-x^2/2}:
#   int g^2 dx        = sum a_n^2 * 2^n n!  sqrt(pi)
#   int g dx          = sum_{n even} a_n * 2^n n!  sqrt(pi)
#   g' = (p' - x p) e^{-x^2/2},  and p'-xp = sum c_k H_k with
#     c_0 = -a_0
#     c_k = 2k a_k - (k+1) a_{k+1} - a_{k-1}   for k >= 1
#   int (g')^2 dx     = sum c_k^2 * 2^k k!  sqrt(pi)
#   int |F g|^2 dx    = int g^2 dx   (Plancherel, FT of H_n g is (-i)^n H_n g)

def _fact2(n):
    return math.factorial(n) * (2.0 ** n)

# basis index -> (n, 2^n n!)
_BASIS = [(0, _fact2(0)), (4, _fact2(4)), (8, _fact2(8)), (12, _fact2(12))]
_NIDX = {n: i for i, (n, _) in enumerate(_BASIS)}
_W = np.array([w for _, w in _BASIS])  # weight 2^n n! for each basis element
_NVAL = np.array([n for n, _ in _BASIS], dtype=float)


def _moments(a):
    """Return (I_g, I_g2, I_gp2) for full 4-vector a = [a0,a4,a8,a12]."""
    I_g = float(np.dot(a, _W))                 # int g dx (only even n contribute; all are even)
    I_g2 = float(np.dot(a * a, _W))            # int g^2 dx
    # momentum operator: since no two basis degrees differ by 1, c_k = 2*n_k*a_k
    c = 2.0 * _NVAL * a
    I_gp2 = float(np.dot(c * c, _W))
    return I_g, I_g2, I_gp2


def _bound_and_grad(c):
    """Exact bound value and analytic gradient w.r.t. (a0,a4,a8)."""
    a0, a4, a8 = c
    a12 = -(a0 * 1.0 + a4 * (-12.0) + a8 * 168.0) / (-4960.0)
    a = np.array([a0, a4, a8, a12])

    I_g, I_g2, I_gp2 = _moments(a)
    I_g2 = max(I_g2, 1e-300)
    I_gp2 = max(I_gp2, 1e-300)

    # val = |I_g| * sqrt(I_gp2) * sqrt(I_g2)   (I_g2 == int|Fg|^2)
    sign = 1.0 if I_g >= 0.0 else -1.0
    ag = abs(I_g)
    val = ag * math.sqrt(I_gp2) * math.sqrt(I_g2)

    # d/d a_j  (ag * sqrt(I_gp2) * sqrt(I_g2))
    dI_g = _W                       # dI_g/d a_j = 2^n n!
    dI_g2 = 2.0 * a * _W
    c_mom = 2.0 * _NVAL * a
    dI_gp2 = 2.0 * c_mom * _W

    sgp = 0.5 / math.sqrt(I_gp2)
    sg2 = 0.5 / math.sqrt(I_g2)

    grad_full = sign * (dI_g * math.sqrt(I_gp2) * math.sqrt(I_g2)
                        + ag * sgp * dI_gp2 * math.sqrt(I_g2)
                        + ag * math.sqrt(I_gp2) * sg2 * dI_g2)
    grad = grad_full[:3]
    return val, grad


def _objective(c):
    return _bound_and_grad(c)[0]


def _objective_grad(c):
    return _bound_and_grad(c)


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    try:
        best_val = _objective(best_coeffs)
    except Exception:
        return best_coeffs

    def _try(x0):
        nonlocal best_val, best_coeffs
        if time.time() > _DEADLINE - _MARGIN:
            return
        try:
            res = minimize(
                _objective_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Dense deterministic grid around the seed
    rng = np.random.default_rng(12345)
    # Coarse-to-fine grid
    for radius in (0.30, 0.15, 0.075, 0.0375):
        if time.time() > _DEADLINE - _MARGIN:
            break
        for d0 in np.linspace(-radius, radius, 7):
            for d4 in np.linspace(-radius, radius, 7):
                for d8 in np.linspace(-radius, radius, 7):
                    x0 = best_coeffs + np.array([d0, d4, d8])
                    _try(x0)

    # Random multi-start
    for _ in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break
        x0 = best_coeffs + rng.normal(0, 0.10, size=3)
        _try(x0)

    # Final tight polish from current best
    _try(best_coeffs.copy())
    if time.time() < _DEADLINE - _MARGIN:
        _try(best_coeffs + rng.normal(0, 1e-3, size=3))

    # High-precision final polish with much tighter tolerances
    if time.time() < _DEADLINE - _MARGIN:
        for _ in range(5):
            if time.time() > _DEADLINE - _MARGIN:
                break
            try:
                res = minimize(
                    _objective_grad, best_coeffs, jac=True, method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-18, 'gtol': 1e-15},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
                else:
                    break
            except Exception:
                break

    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return best_coeffs
