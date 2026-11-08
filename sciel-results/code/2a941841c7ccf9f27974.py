import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize, differential_evolution
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600  # seconds
    
    def laguerre_norm_sq(roots):
        """Compute squared Laguerre norm of P(x) = prod (x - r_i)^2"""
        # Build polynomial as product of (x - r_i)^2 for each root
        # Start with polynomial [1] (constant 1)
        poly = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2r*x + r^2
            factor = np.array([r*r, -2*r, 1.0])
            poly = np.convolve(poly, factor)
        
        # Convert from standard basis to Laguerre basis
        # Use laguerre.poly2lag
        lag_coeffs = laguerre.poly2lag(poly)
        
        # The squared norm in Laguerre inner product:
        # For Laguerre polynomials, <L_m, L_n> = delta_mn
        # So ||P||^2 = sum(c_k^2) where P = sum c_k L_k
        norm_sq = np.sum(lag_coeffs**2)
        return norm_sq
    
    def cost_func(s_vars):
        """Cost function: s_vars are sqrt of roots, so roots = s_vars^2"""
        roots = s_vars**2
        try:
            return laguerre_norm_sq(roots)
        except Exception:
            return 1e10
    
    # Initial guess: use Laguerre-Gauss nodes as starting point
    nodes, _ = laguerre.laggauss(12)
    s_init = np.sqrt(nodes)
    
    # Try local optimization first with multiple restarts
    best_val = float('inf')
    best_s = s_init.copy()
    
    # Local optimization with L-BFGS-B
    try:
        result = minimize(cost_func, s_init, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_s = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 60:
        return [float(x**2) for x in best_s]
    
    # Try global optimization with differential evolution
    # Bounds for s_vars: roots should be positive, reasonable range
    # Laguerre-Gauss nodes for n=12 are roughly in [0.1, 40]
    # So s_vars in [0.01, 10]
    lower_bounds = [0.01] * 12
    upper_bounds = [10.0] * 12
    
    try:
        de_result = differential_evolution(cost_func, 
                                           bounds=list(zip(lower_bounds, upper_bounds)),
                                           seed=42, maxiter=200, tol=1e-8,
                                           polish=True, init='sobol')
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_s = de_result.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 30:
        return [float(x**2) for x in best_s]
    
    # Final local polish
    try:
        result = minimize(cost_func, best_s, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_s = result.x
    except Exception:
        pass
    
    # Return the roots (squared values of s_vars)
    roots = [float(x**2) for x in best_s]
    
    # Verify all positive
    for r in roots:
        if r <= 0:
            roots = [float(x**2) for x in s_init]
            break
    
    return roots
# EVOLVE-BLOCK-END
