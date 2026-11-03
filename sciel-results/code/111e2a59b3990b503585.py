"""Rewritten program using Nelder-Mead simplex optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import math


def _hermite_polynomial(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    # Use recurrence relation: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2 * x
    
    H_prev = np.ones_like(x, dtype=float)  # H_0
    H_curr = 2 * x  # H_1
    
    for k in range(1, n):
        H_next = 2 * x * H_curr - 2 * k * H_prev
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


def _upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x) where c12 is chosen
    so g(0) = 0.
    
    A(f) = ||f||_1 / ||f||_2
    The product A(f) * A(F(f)) gives the uncertainty bound.
    """
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    # Use a sufficiently large grid for accurate integration
    # The function decays like x^12, so we need to capture the tails
    x = np.linspace(-15, 15, 8192)  # Power of 2 for FFT efficiency
    H0 = _hermite_polynomial(0, x)
    H4 = _hermite_polynomial(4, x)
    H8 = _hermite_polynomial(8, x)
    H12 = _hermite_polynomial(12, x)
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # Check if g is essentially zero
    if np.max(np.abs(g)) < 1e-15:
        return 1e10  # Invalid
    
    # Compute L1 and L2 norms using trapezoidal rule for better accuracy
    dx = x[1] - x[0]
    L1 = np.trapz(np.abs(g), x)
    L2 = np.sqrt(np.trapz(g**2, x))
    
    if L2 < 1e-15:
        return 1e10
    
    A_g = L1 / L2
    
    # Compute Fourier transform using FFT with proper scaling
    # F(ξ) = ∫ g(x) exp(-2πi x ξ) dx
    # FFT gives: sum g[n] exp(-2πi kn/N)
    # So F(ξ_k) ≈ dx * FFT(g)[k] where ξ_k = k/(N*dx)
    
    from scipy.fft import fft, fftshift, fftfreq
    
    G = fft(g) * dx  # Scale by dx for integration
    
    # Frequency spacing: dξ = 1/(N*dx)
    # L1 norm of G: sum |G[k]| * dξ
    dxi = 1.0 / (len(x) * dx)
    
    L1_G = np.sum(np.abs(G)) * dxi
    L2_G = np.sqrt(np.sum(np.abs(G)**2)) * dxi
    
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
    # Try multiple initial guesses to avoid local minima
    initial_guesses = [
        np.array([-1.13, 0.04, 0.000308]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0004]),
        np.array([-0.5, 0.06, 0.0002]),
        np.array([-2.0, 0.02, 0.0005]),
        np.array([-1.13, 0.04, 0.000308]),  # Original
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
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'maxiter': 3000,
                    'maxfev': 3000,
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
