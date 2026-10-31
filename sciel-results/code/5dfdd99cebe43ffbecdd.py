"""Optimized uncertainty inequality coefficient search with log-magnitude parameterization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    Hn_minus_1 = 2 * x
    Hn_minus_2 = np.ones_like(x)
    for k in range(2, n + 1):
        Hn = 2 * x * Hn_minus_1 - 2 * (k - 1) * Hn_minus_2
        Hn_minus_2 = Hn_minus_1
        Hn_minus_1 = Hn
    return Hn_minus_1


def _get_coeffs(c0, c4, c8):
    """Get all four coefficients with c12 determined by g(0)=0."""
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -30240.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    return np.array([c0, c4, c8, c12])


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function.
    Uses numerical integration on a fine grid.
    """
    coeffs = _get_coeffs(c0, c4, c8)
    
    x = np.linspace(-20, 20, 40001)
    dx = x[1] - x[0]
    
    # Compute g(x) = sum c_n H_n(x) exp(-x^2/2)
    g = np.zeros_like(x)
    for n, c in [(0, coeffs[0]), (4, coeffs[1]), (8, coeffs[2]), (12, coeffs[3])]:
        g += c * _hermite(n, x) * np.exp(-x**2 / 2)
    
    l1_g = np.sum(np.abs(g)) * dx
    
    # Fourier transform: g_hat(ξ) = sum c_n (i)^n H_n(ξ) exp(-ξ^2/2)
    g_hat = np.zeros_like(x)
    for n, c in [(0, coeffs[0]), (4, coeffs[1]), (8, coeffs[2]), (12, coeffs[3])]:
        g_hat += c * (1j)**n * _hermite(n, x) * np.exp(-x**2 / 2)
    
    l1_g_hat = np.sum(np.abs(g_hat)) * dx
    
    return l1_g * l1_g_hat


def construct():
    """
    Find optimal coefficients using log-magnitude parameterization
    and differential evolution with local refinement.
    """
    deadline = time.time() + 195
    
    # Known good starting point
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Log-magnitude parameterization: c_i = sign_i * exp(log_mag_i)
    # This decouples scale from sign for better exploration
    def log_param_to_coeffs(lp):
        s0, lm0, s4, lm4, s8, lm8 = lp
        c0 = s0 * np.exp(lm0)
        c4 = s4 * np.exp(lm4)
        c8 = s8 * np.exp(lm8)
        return c0, c4, c8
    
    def log_param_bound(lp):
        c0, c4, c8 = log_param_to_coeffs(lp)
        return _compute_bound(c0, c4, c8)
    
    # Initial log parameters from x0
    def coeffs_to_log_params(c0, c4, c8):
        s0 = np.sign(c0) if c0 != 0 else 1.0
        s4 = np.sign(c4) if c4 != 0 else 1.0
        s8 = np.sign(c8) if c8 != 0 else 1.0
        lm0 = np.log(np.abs(c0)) if c0 != 0 else -10
        lm4 = np.log(np.abs(c4)) if c4 != 0 else -10
        lm8 = np.log(np.abs(c8)) if c8 != 0 else -10
        return np.array([s0, lm0, s4, lm4, s8, lm8])
    
    lp0 = coeffs_to_log_params(*x0)
    
    # Differential evolution in log-parameter space
    if time.time() < deadline - 40:
        try:
            # Bounds for log parameters
            bounds = [
                [-1, 1],  # sign c0
                [-10, 2],  # log|c0|
                [-1, 1],  # sign c4
                [-10, 2],  # log|c4|
                [-1, 1],  # sign c8
                [-10, 2],  # log|c8|
            ]
            
            result = differential_evolution(
                log_param_bound,
                bounds,
                seed=42,
                maxiter=30,
                popsize=20,
                tol=1e-10,
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_score:
                c0, c4, c8 = log_param_to_coeffs(result.x)
                best_score = result.fun
                best_x = np.array([c0, c4, c8])
        except Exception:
            pass
    
    # Local refinement in original parameter space from multiple starts
    start_points = [
        best_x,
        x0,
    ]
    
    # Add some perturbations
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        pert = best_x + 0.1 * np.random.randn(3)
        start_points.append(pert)
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(
                lambda x: _compute_bound(*x),
                sp,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
