"""Optimized program for uncertainty inequality - tune Hermite coefficients to minimize C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hermite_h(n, x):
    """Compute Hermite polynomial H_n(x) using three-term recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _objective(a0, a4, a8):
    """
    Compute the uncertainty bound C4 for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is fixed by g(0) = 0.
    
    The bound involves the product of L2 norms of x*g and g' relative to ||g||_2^2.
    """
    # H_k(0) for even k
    # H_0(0)=1, H_4(0)=3, H_8(0)=3, H_12(0)=210
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 3.0) / 210.0
    
    # Numerical integration
    x = np.linspace(-10, 10, 30001)
    dx = x[1] - x[0]
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    
    # Check if g is essentially zero
    norm_g_sq = np.trapz(g**2, x)
    if norm_g_sq < 1e-15:
        return 1e10
    
    # Compute ||x*g||_2
    xg = x * g
    norm_xg_sq = np.trapz(xg**2, x)
    
    # Compute ||g'||_2 using numerical differentiation
    dg = np.gradient(g, dx)
    norm_dg_sq = np.trapz(dg**2, x)
    
    # Uncertainty bound: C = ||x*g||_2 * ||g'||_2 / ||g||_2^2
    # This is the constant in ||x*f||_2 * ||f'||_2 >= C * ||f||_2^2
    C_bound = np.sqrt(max(norm_xg_sq, 0) * max(norm_dg_sq, 0)) / norm_g_sq
    
    return C_bound

def construct():
    """
    Find coefficients (a0, a4, a8) minimizing the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    # Seed value
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_x = x0.copy()
    best_val = _objective(*x0)
    
    # Multi-start local optimization
    starts = [
        x0,
        np.array([-1.15, 0.04, 0.0003]),
        np.array([-1.05, 0.045, 0.0005]),
        np.array([-1.25, 0.035, 0.0002]),
        np.array([-1.10, 0.042, 0.00035]),
        np.array([-1.00, 0.050, 0.0006]),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(
                lambda x: _objective(*x),
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Global optimization if time permits
    if time.time() < deadline - 80:
        try:
            bounds = [(-3.0, 0.0), (-0.3, 0.3), (-0.02, 0.02)]
            de_res = differential_evolution(
                lambda x: _objective(*x),
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-12,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 15:
        try:
            polish = minimize(
                lambda x: _objective(*x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if polish.fun < best_val:
                best_val = polish.fun
                best_x = polish.x.copy()
        except Exception:
            pass
    
    return np.array(best_x, dtype=float)
# EVOLVE-BLOCK-END
