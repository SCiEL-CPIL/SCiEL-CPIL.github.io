"""Rewritten program: optimize Hermite coefficients to minimize upper bound C4."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 200  # Safety margin before 210s kill


def _elapsed():
    return time.time() - _START_TIME


def _hermite_poly(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(1, n):
        h2 = 2 * x * h1 - 2 * k * h0
        h0, h1 = h1, h2
    return h1


def _compute_c12(c0, c4, c8):
    """Compute c12 from constraint g(0) = 0."""
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    return -(c0 + 3 * c4 + 105 * c8) / 10395.0


def _objective(coeffs):
    """Compute the upper bound C4 from coefficients c0, c4, c8."""
    if _elapsed() > _TIME_LIMIT:
        return 1e10  # Timeout guard
    
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # Compute the function g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # and its Fourier transform, then compute some measure of the product
    
    # Use a grid for numerical computation
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    
    h0 = _hermite_poly(0, x)
    h4 = _hermite_poly(4, x)
    h8 = _hermite_poly(8, x)
    h12 = _hermite_poly(12, x)
    
    poly = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    g = poly * np.exp(-x**2 / 2)
    
    # Compute L2 norm of g
    norm_g = np.sqrt(np.sum(g**2) * dx)
    
    # Compute Fourier transform numerically
    # For Hermite functions, the FT is known, but let's compute numerically
    # Use FFT with appropriate scaling
    N = len(x)
    g_fft = np.fft.fft(g) * dx
    # Shift to center
    g_fft = np.fft.fftshift(g_fft)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    
    # Compute L2 norm of Fourier transform
    norm_gf = np.sqrt(np.sum(np.abs(g_fft)**2) * (2 * np.pi / N))
    
    # A proxy for the upper bound: product of norms or some related quantity
    # Use a combination that captures the "spread"
    # Compute moment-based measures
    x2 = x**2
    moment_g = np.sum(x2 * g**2) * dx / (norm_g**2 + 1e-15)
    
    freq2 = freq**2
    moment_gf = np.sum(freq2 * np.abs(g_fft)**2) * (2 * np.pi / N) / (norm_gf**2 + 1e-15)
    
    # Upper bound proxy: product of spreads
    c4_bound = np.sqrt(moment_g) * np.sqrt(moment_gf)
    
    return c4_bound


def construct():
    """Return three coefficients (c0, c4, c8) that minimize the upper bound C4."""
    if _elapsed() > _TIME_LIMIT:
        # Return prior state of the art
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Start from the prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = x0
    best_score = _objective(x0)
    
    # Try SLSQP optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if result.success and _elapsed() < _TIME_LIMIT:
            if result.fun < best_score:
                best_result = result.x
                best_score = result.fun
    except Exception:
        pass
    
    # Try Nelder-Mead as backup
    if _elapsed() < _TIME_LIMIT - 30:
        try:
            result2 = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if _elapsed() < _TIME_LIMIT:
                if result2.fun < best_score:
                    best_result = result2.x
                    best_score = result2.fun
        except Exception:
            pass
    
    # Try a few random perturbations
    if _elapsed() < _TIME_LIMIT - 20:
        np.random.seed(42)
        for _ in range(20):
            if _elapsed() > _TIME_LIMIT - 10:
                break
            perturbation = np.random.normal(0, 0.1, size=3)
            candidate = x0 + perturbation
            score = _objective(candidate)
            if score < best_score:
                best_result = candidate
                best_score = score
    
    return best_result
# EVOLVE-BLOCK-END
