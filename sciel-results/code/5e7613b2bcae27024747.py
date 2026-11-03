"""Rewritten program using Nelder-Mead simplex optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft
import math


def _hermite_polynomial(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    
    H_prev = np.ones_like(x, dtype=float)  # H_0
    H_curr = 2.0 * x  # H_1
    
    for k in range(1, n):
        H_next = 2.0 * x * H_curr - 2.0 * k * H_prev
        H_prev = H_curr
        H_curr = H_next
    
    return H_curr


def _compute_coefficients(c0, c4, c8):
    """
    Given three free coefficients for H0, H4, H8, compute the H12 coefficient
    such that g(0) = 0.
    
    g(x) = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
    
    At x=0:
    H0(0) = 1
    H4(0) = -12
    H8(0) = 1008
    H12(0) = -103950
    
    So: c0 * 1 + c4 * (-12) + c8 * 1008 + c12 * (-103950) = 0
    => c12 = (c0 - 12*c4 + 1008*c8) / 103950
    """
    c12 = (c0 - 12.0 * c4 + 1008.0 * c8) / 103950.0
    return c0, c4, c8, c12


# Precompute grid and basis polynomials for efficiency
_N = 16384  # Power of 2 for FFT efficiency, larger for accuracy
_X = np.linspace(-20.0, 20.0, _N)
_DX = _X[1] - _X[0]
_DXI = 1.0 / (_N * _DX)

_H0 = _hermite_polynomial(0, _X)
_H4 = _hermite_polynomial(4, _X)
_H8 = _hermite_polynomial(8, _X)
_H12 = _hermite_polynomial(12, _X)


def _upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x) where c12 is chosen
    so g(0) = 0.
    
    A(f) = ||f||_1 / ||f||_2
    The product A(f) * A(F(f)) gives the uncertainty bound.
    """
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    g = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    
    # Check if g is essentially zero
    max_abs_g = np.max(np.abs(g))
    if max_abs_g < 1e-15:
        return 1e10  # Invalid
    
    # Compute L1 and L2 norms using trapezoidal rule for better accuracy
    L1 = np.trapz(np.abs(g), _X)
    L2_sq = np.trapz(g * g, _X)
    L2 = np.sqrt(L2_sq)
    
    if L2 < 1e-15:
        return 1e10
    
    A_g = L1 / L2
    
    # Compute Fourier transform using FFT with proper scaling
    # F(xi) = integral g(x) exp(-2 pi i x xi) dx
    # FFT gives: sum g[n] exp(-2 pi i k n / N)
    # So F(xi_k) approx _DX * FFT(g)[k] where xi_k = k / (N * _DX)
    
    G = fft(g) * _DX
    
    L1_G = np.sum(np.abs(G)) * _DXI
    L2_G = np.sqrt(np.sum(np.abs(G) ** 2)) * _DXI
    
    if L2_G < 1e-15:
        return 1e10
    
    A_G = L1_G / L2_G
    
    # The upper bound is A(g) * A(G)
    bound = A_g * A_G
    
    return bound


def _objective(params):
    """Objective function for optimization: minimize the upper bound."""
    c0, c4, c8 = params
    try:
        val = _upper_bound_C4(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound
    on A(f) * A(Fourier(f)) using Nelder-Mead simplex optimization.
    """
    # Multiple initial guesses to avoid local minima
    initial_guesses = [
        np.array([-1.13, 0.04, 0.000308]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0004]),
        np.array([-0.5, 0.06, 0.0002]),
        np.array([-2.0, 0.02, 0.0005]),
        np.array([-1.2, 0.035, 0.00035]),
        np.array([-0.8, 0.045, 0.00025]),
    ]
    
    best_result = None
    best_fun = 1e10
    
    for x0 in initial_guesses:
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'adaptive': True
                }
            )
            
            if np.isfinite(result.fun) and result.fun < best_fun:
                best_fun = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None:
        return np.array(best_result)
    else:
        # Fallback to original initial guess
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
