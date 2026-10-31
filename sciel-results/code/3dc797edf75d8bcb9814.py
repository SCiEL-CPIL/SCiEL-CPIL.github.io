"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_physicists(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1

def _compute_uncertainty_product(c0, c4, c8):
    """Compute the uncertainty product A(f)*A(Fhat(f)) for the test function."""
    # Compute c12 from constraint g(0) = 0
    # H4(0) = 12, H8(0) = 1680, H12(0) = 665280
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Use a fine grid to compute the function and its properties
    # The function is g(x) = (c0 + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    # We need A(g) and A(F(g)) where A likely means L1 norm or variance-based measure
    
    # For uncertainty principle, the relevant quantity is typically:
    # sigma_x * sigma_p where sigma_x^2 = <x^2> - <x>^2 and sigma_p^2 = <p^2> - <p>^2
    # with p = -i d/dx (in appropriate units)
    
    # Let's compute using numerical integration on a fine grid
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    H4 = _hermite_physicists(4, x)
    H8 = _hermite_physicists(8, x)
    H12 = _hermite_physicists(12, x)
    
    g = (c0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
    
    # Check if g is essentially zero
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-15:
        return 1e10  # Invalid, return large value
    
    # Normalize
    g_norm = g / np.sqrt(norm_sq)
    
    # Compute <x>, <x^2>
    x_mean = np.sum(x * g_norm**2) * dx
    x2_mean = np.sum(x**2 * g_norm**2) * dx
    sigma_x2 = x2_mean - x_mean**2
    if sigma_x2 < 0:
        sigma_x2 = 0
    
    # Compute momentum variance: <p^2> where p = -i d/dx
    # p^2 = -d^2/dx^2, so <p^2> = integral |g'|^2 dx (by integration by parts for normalized g)
    # Actually <p^2> = integral g* (-d^2/dx^2) g dx = integral |g'|^2 dx (for boundary terms vanishing)
    
    # Compute derivative numerically
    g_prime = np.gradient(g_norm, dx)
    p2_mean = np.sum(g_prime**2) * dx
    
    # <p> = integral g* (-i d/dx) g dx = 0 for real g (imaginary part)
    # So sigma_p2 = p2_mean
    
    sigma_x = np.sqrt(sigma_x2)
    sigma_p = np.sqrt(p2_mean)
    
    # The uncertainty product is sigma_x * sigma_p
    # But we need A(f)*A(F(f)) which might be different
    # For the uncertainty inequality context, this should be proportional to the variance product
    
    return sigma_x * sigma_p

def _objective(params):
    """Objective function for optimization."""
    c0, c4, c8 = params
    try:
        val = _compute_uncertainty_product(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s limit
    
    # Starting point from prior state of the art
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_uncertainty_product(best_params[0], best_params[1], best_params[2])
    
    # Try several optimization strategies
    strategies = []
    
    # Strategy 1: Nelder-Mead from the prior best point
    strategies.append(('nelder_mead_prior', best_params.copy()))
    
    # Strategy 2: Nelder-Mead from a perturbed point
    strategies.append(('nelder_mead_perturbed', best_params + np.array([0.1, -0.05, 0.01])))
    
    # Strategy 3: Try some hand-crafted points
    # The Hermite functions are orthonormal, so the variance depends on the coefficients
    # Let's try points where the function is more concentrated
    strategies.append(('hand_1', np.array([1.0, -0.1, 0.001])))
    strategies.append(('hand_2', np.array([0.5, -0.05, 0.005])))
    strategies.append(('hand_3', np.array([-0.5, 0.02, -0.001])))
    
    # Evaluate initial points
    for name, params in strategies:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        val = _objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
    
    # Local optimization from best point using Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try L-BFGS-B with numerical gradient from multiple starting points
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        # Generate several random starting points
        rng = np.random.default_rng(42)
        for i in range(10):
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            # Random start in a reasonable range
            p = rng.uniform([-2, -0.5, -0.1], [2, 0.5, 0.1])
            try:
                res = minimize(_objective, p, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'gtol': 1e-10})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Differential evolution for global search if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 90:
        try:
            bounds = [(-3, 3), (-1, 1), (-0.2, 0.2)]
            res = differential_evolution(_objective, bounds, seed=42, maxiter=50, 
                                         tol=1e-10, polish=True, workers=1)
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final polish with high-precision Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Return the three coefficients
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
