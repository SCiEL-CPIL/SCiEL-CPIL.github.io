"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in descending powers of x."""
    H0 = np.array([1.0])
    H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    H8 = np.array([256.0, 0.0, -1280.0, 0.0, 2880.0, 0.0, -2880.0, 0.0, 168.0])
    H12 = np.array([4096.0, 0.0, -38400.0, 0.0, 126720.0, 0.0, -215040.0, 0.0,
                     193536.0, 0.0, -80640.0, 0.0, 10368.0])
    return H0, H4, H8, H12

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients (descending powers) at x."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        result += c * (x ** power)
    return result

def _compute_c4_upper_bound(a0, a4, a8):
    """Compute the upper bound on C4 for given coefficients.
    
    The C4 upper bound is related to the product A(f)*A(Fourier(f)),
    where A(f) is a measure of the spread/amplitude of f.
    For this task, we interpret it as the uncertainty product 
    (standard deviation in x) * (standard deviation in k)
    for the normalized test function g(x) = P(x) * exp(-x^2/2).
    """
    H0, H4, H8, H12 = _hermite_coeffs()
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 10368.0
    
    # Use a finer grid for better accuracy
    x = np.linspace(-12, 12, 30000)
    dx = x[1] - x[0]
    
    H0_vals = _poly_eval(H0, x)
    H4_vals = _poly_eval(H4, x)
    H8_vals = _poly_eval(H8, x)
    H12_vals = _poly_eval(H12, x)
    
    g = (a0 * H0_vals + a4 * H4_vals + a8 * H8_vals + a12 * H12_vals) * np.exp(-x**2 / 2.0)
    
    # Normalize g
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-15:
        return 1e10
    norm = np.sqrt(norm_sq)
    g = g / norm
    
    # Position variance
    x_mean = np.sum(x * g**2) * dx
    x2_mean = np.sum(x**2 * g**2) * dx
    var_x = x2_mean - x_mean**2
    
    if var_x < 1e-15:
        return 1e10
    
    # Momentum variance via derivative
    dg_dx = np.gradient(g, dx)
    p2_mean = np.sum(dg_dx**2) * dx
    p_mean = 0.0  # For real, decaying functions
    var_p = p2_mean - p_mean**2
    
    if var_p < 1e-15:
        return 1e10
    
    return var_x * var_p

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
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
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(_compute_c4_upper_bound, sp, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_compute_c4_upper_bound, best_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
