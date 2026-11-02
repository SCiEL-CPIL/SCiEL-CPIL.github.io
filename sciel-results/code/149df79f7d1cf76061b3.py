"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial values on a fixed grid for speed
_N_GRID = 65536  # Power of 2 for FFT efficiency
_X_MIN = -12.0
_X_MAX = 12.0
_X = np.linspace(_X_MIN, _X_MAX, _N_GRID)
_DX = _X[1] - _X[0]
# FFT frequency vector for derivative
_K = 2.0 * np.pi * np.fft.fftfreq(_N_GRID, d=_DX)
_GAUSS = np.exp(-_X**2 / 2.0)
_X2 = _X**2

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in descending powers of x."""
    H0 = np.array([1.0])
    H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    H8 = np.array([256.0, 0.0, -1280.0, 0.0, 2880.0, 0.0, -2880.0, 0.0, 168.0])
    H12 = np.array([4096.0, 0.0, -38400.0, 0.0, 126720.0, 0.0, -215040.0, 0.0,
                     193536.0, 0.0, -80640.0, 0.0, 10368.0])
    return H0, H4, H8, H12

# Precompute Hermite values on the grid
_H0, _H4, _H8, _H12 = _hermite_coeffs()
_H0_V = np.polynomial.polynomial.polyval(_X, np.array([1.0]))
_H4_V = np.polynomial.polynomial.polyval(_X, np.array([12.0, 0.0, -48.0, 0.0, 16.0]))
_H8_V = np.polynomial.polynomial.polyval(_X, np.array([168.0, 0.0, -2880.0, 0.0, 2880.0, 0.0, -1280.0, 0.0, 256.0]))
_H12_V = np.polynomial.polynomial.polyval(_X, np.array([10368.0, 0.0, -80640.0, 0.0, 193536.0, 0.0, -215040.0, 0.0, 126720.0, 0.0, -38400.0, 0.0, 4096.0]))

# Precompute the Gaussian-weighted Hermite values
_H0_G = _H0_V * _GAUSS
_H4_G = _H4_V * _GAUSS
_H8_G = _H8_V * _GAUSS
_H12_G = _H12_V * _GAUSS

def _compute_c4_upper_bound(a0, a4, a8):
    """Compute the upper bound on C4 for given coefficients.
    
    Uses FFT-based differentiation for accurate momentum variance.
    """
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 10368.0
    
    # Build g(x) = P(x) * exp(-x^2/2)
    g = a0 * _H0_G + a4 * _H4_G + a8 * _H8_G + a12 * _H12_G
    
    # Normalize g
    norm_sq = np.sum(g**2) * _DX
    if norm_sq < 1e-15:
        return 1e10
    norm = np.sqrt(norm_sq)
    g = g / norm
    
    # Position variance
    g2 = g**2
    x_mean = np.sum(_X * g2) * _DX
    x2_mean = np.sum(_X2 * g2) * _DX
    var_x = x2_mean - x_mean**2
    
    if var_x < 1e-15:
        return 1e10
    
    # Momentum variance via FFT differentiation (much more accurate than np.gradient)
    g_hat = np.fft.fft(g)
    dg_hat = 1j * _K * g_hat
    dg_dx = np.fft.ifft(dg_hat).real
    p2_mean = np.sum(dg_dx**2) * _DX
    var_p = p2_mean  # p_mean = 0 for real, decaying functions
    
    if var_p < 1e-15:
        return 1e10
    
    return var_x * var_p

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    start_time = time.time()
    deadline = start_time + 205.0  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_score = _compute_c4_upper_bound(*seed)
    
    # Multiple starting points for multi-start Nelder-Mead
    start_points = [
        seed,
        seed + np.array([0.05, 0.005, 0.0005]),
        seed - np.array([0.05, 0.005, 0.0005]),
        seed + np.array([0.2, 0.02, 0.002]),
        seed - np.array([0.2, 0.02, 0.002]),
        np.array([-1.0, 0.04, 0.0004]),
        np.array([-0.8, 0.032, 0.00032]),
        np.array([-1.2, 0.048, 0.00048]),
        np.array([-0.9, 0.036, 0.00036]),
        np.array([-1.1, 0.044, 0.00044]),
    ]
    
    # Multi-start Nelder-Mead optimization
    for i, sp in enumerate(start_points):
        if time.time() > deadline - 25.0:
            break
        try:
            result = minimize(_compute_c4_upper_bound, sp, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with even tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_compute_c4_upper_bound, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
