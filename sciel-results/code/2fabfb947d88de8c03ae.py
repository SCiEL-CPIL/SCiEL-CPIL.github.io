"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Strategy: use analytic L2 norms via orthogonality, vectorized L1 norms on a fine grid,
and L-BFGS-B with finite-difference gradients.  The P(0)=0 constraint fixes c12
analytically, leaving three free coefficients.
"""

import numpy as np
from scipy import optimize
from scipy.special import factorial
import time

# Hermite values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1190.0
_H12_0 = 113880.0

# Precompute L2 norm factors: sqrt(pi) * 2^n * n! for n in {0,4,8,12}
_DEGREES = [0, 4, 8, 12]
_L2_FACTORS = np.array([
    np.sqrt(np.pi) * (2.0 ** d) * float(factorial(d, exact=True))
    for d in _DEGREES
])

# Precompute FT prefactors: sqrt(2*pi) * (2*pi)^n * (-i)^n
_TWO_PI = 2.0 * np.pi
_FT_PREFACTORS = np.array([
    np.sqrt(_TWO_PI) * (_TWO_PI ** d) * ((-1j) ** d)
    for d in _DEGREES
])


def _hermite_h(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)
    if n == 0:
        h = np.ones_like(x)
    elif n == 1:
        h = 2.0 * x
    else:
        h0 = np.ones_like(x)
        h1 = 2.0 * x
        for k in range(2, n + 1):
            h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
            h0, h1 = h1, h2
        h = h1
    if scalar:
        return h.item()
    return h


def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    return -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0


# Precompute grids and Hermite values for fast evaluation
_N_PTS = 4000
_X_MIN = -6.0
_X_MAX = 6.0
_X_GRID = np.linspace(_X_MIN, _X_MAX, _N_PTS)
_DX = _X_GRID[1] - _X_GRID[0]

# g(x) = [c0*H0 + c4*H4 + c8*H8 + c12*H12] * exp(-x^2/2)
_H_G = np.array([_hermite_h(d, _X_GRID) for d in _DEGREES])  # shape (4, N)
_GAUSS_G = np.exp(-_X_GRID**2 / 2.0)
_H_G_WEIGHTED = _H_G * _GAUSS_G[np.newaxis, :]  # (4, N)

# F(g)(k) = sum_n c_n * ft_prefactor_n * H_n(2*pi*k) * exp(-(2*pi)^2 * k^2)
_H_FT = np.array([_hermite_h(d, _TWO_PI * _X_GRID) for d in _DEGREES])  # (4, N)
_GAUSS_FT = np.exp(-(_TWO_PI**2) * _X_GRID**2)
_H_FT_WEIGHTED = _H_FT * _GAUSS_FT[np.newaxis, :]  # (4, N)


def _bound_fast(coeffs):
    """
    Fast computation of the upper bound A(f)*A(F(f)).
    coeffs = np.array([c0, c4, c8])
    """
    c12 = _compute_c12(coeffs[0], coeffs[1], coeffs[2])
    full_coeffs = np.array([coeffs[0], coeffs[1], coeffs[2], c12])

    # L2 norm squared (analytic)
    l2_sq = float(np.dot(full_coeffs**2, _L2_FACTORS))
    if l2_sq < 1e-15:
        return 1e10
    l2 = np.sqrt(l2_sq)

    # L1 norm of g: sum |sum_n c_n * H_n(x)*gauss(x)| * dx
    g_vals = full_coeffs @ _H_G_WEIGHTED  # (N,)
    l1_g = float(np.sum(np.abs(g_vals)) * _DX)

    # L1 norm of F(g): sum |sum_n c_n * ft_prefactor_n * H_n(2*pi*k)*gauss(k)| * dx
    # Note: F(g) is complex in general, so take |F(g)(k)|
    fg_vals = np.zeros(_N_PTS, dtype=complex)
    for idx in range(4):
        fg_vals += full_coeffs[idx] * _FT_PREFACTORS[idx] * _H_FT_WEIGHTED[idx]
    l1_Fg = float(np.sum(np.abs(fg_vals)) * _DX)

    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10

    A_f = l1_g / l2
    A_F = l1_Fg / l2  # Plancherel: ||F(g)||_2 = ||g||_2
    return A_f * A_F


def _bound_and_grad(coeffs):
    """Compute the bound and its gradient via central finite differences."""
    val = _bound_fast(coeffs)
    eps = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_minus = coeffs.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        bp = _bound_fast(c_plus)
        bm = _bound_fast(c_minus)
        grad[i] = (bp - bm) / (2.0 * eps)
    return val, grad


def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit

    # Starting points
    x0 = np.array([-1.13, 0.04, 0.000308642])

    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            return _bound_fast(np.asarray(coeffs, dtype=float))
        except Exception:
            return 1e10

    # Initial evaluation
    best_coeffs = x0.copy()
    best_val = objective(x0)

    # L-BFGS-B optimization
    try:
        result = optimize.minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass

    # Nelder-Mead as backup
    elapsed = time.time() - start_time
    if elapsed < deadline - 25.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # One more Nelder-Mead polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs, dtype=float)


if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    val = _bound_fast(coeffs)
    print(f"Upper bound on C4: {val}")
