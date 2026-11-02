"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _compute_variance(a0, a4, a8):
    """
    Compute the variance of f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by f(0) = 0.
    
    The variance is E[x^2] = <x^2> / <1> where <g> = integral of g * |f|^2 / integral of |f|^2
    """
    # Compute a12 from constraint f(0) = 0
    # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 2032128
    a12 = -(a0 + 12*a4 + 1680*a8) / 2032128.0
    
    # Define Hermite polynomials (physicists')
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return (4096*x**12 - 73728*x**10 + 580608*x**8 - 2419200*x**6 
                + 5443200*x**4 - 5443200*x**2 + 2032128)
    
    # Use a fine grid for numerical integration
    x = np.linspace(-15, 15, 2000)
    dx = x[1] - x[0]
    
    P = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    f = P * np.exp(-x**2/2)
    
    f2 = f**2
    norm_sq = np.trapz(f2, x)
    
    if norm_sq < 1e-30:
        return 1e10  # Penalty for zero norm
    
    x2_weighted = np.trapz(x**2 * f2, x)
    variance = x2_weighted / norm_sq
    
    return variance


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4. This is related to the uncertainty product.
    For a function that is an eigenfunction of Fourier transform (eigenvalue 1),
    the product of spreads is Var(x)^2. We minimize this.
    """
    return _compute_variance(a0, a4, a8)


def construct():
    start_time = time.time()
    deadline = 210.0  # seconds - evaluator kills at 210s
    safety_margin = 15.0
    effective_deadline = deadline - safety_margin
    
    # Seed values
    seed = np.array([-1.13, 0.04, 0.00030864])
    best_x = seed.copy()
    best_val = _compute_bound(*seed)
    
    # Try multiple starting points with Nelder-Mead
    random_starts = [
        [-1.0, 0.05, 0.0003],
        [-1.2, 0.03, 0.0004],
        [-0.9, 0.06, 0.0002],
        [-1.5, 0.02, 0.0005],
        [-0.8, 0.08, 0.0001],
        [-1.13, 0.04, 0.00030864],
    ]
    
    for x0 in random_starts:
        if time.time() > effective_deadline:
            break
        
        try:
            result = minimize(_compute_bound, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if time.time() < effective_deadline - 30:
        try:
            bounds = [(-3.0, 0.0), (-0.5, 0.5), (-0.001, 0.001)]
            de_result = differential_evolution(_compute_bound, bounds,
                                               maxiter=100, tol=1e-10,
                                               popsize=15, seed=42,
                                               polish=True)
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Final local refinement from best point
    if time.time() < effective_deadline:
        try:
            final_result = minimize(_compute_bound, best_x, method='Nelder-Mead',
                                   options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14,
                                           'adaptive': True})
            if final_result.fun < best_val:
                best_val = final_result.fun
                best_x = final_result.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
