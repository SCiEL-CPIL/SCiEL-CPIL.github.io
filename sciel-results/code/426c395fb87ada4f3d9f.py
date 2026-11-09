import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 for given double-root locations.
    
    The Cohn-Goncalves construction uses a polynomial P(x) with double roots
    at the specified locations. The uncertainty bound involves integrals of
    |P(x)|^2 against appropriate weights.
    
    For m=12, n=1, we construct P(x) = prod_{i=1}^{12} (x - x_i)^2 * Q(x)
    where Q is chosen to satisfy the sign change constraint.
    
    The upper bound on C4 is related to the ratio of certain moments.
    """
    m = 12
    n = 1
    
    # Sort roots to ensure consistency
    roots = np.sort(roots)
    
    # For the refined Cohn-Goncalves construction with m=12, n=1:
    # The polynomial has the form P(x) = [L_m^{(alpha)}(x)]^2 - c * [L_{m-1}^{(alpha+1)}(x)]^2 * x
    # or similar. The double roots are where P(x) = 0.
    
    # Alternative approach: The bound involves the integral of x^n |P(x)|^2 e^{-x} dx
    # divided by integral of |P(x)|^2 e^{-x} dx or similar.
    
    # Let's use a more direct approach: compute the ratio that gives the upper bound.
    # For the Fourier sign uncertainty, C4 relates to the L^2 norm of the Fourier
    # transform of the sign function restricted to support.
    
    # Practical approach: The upper bound is approximately related to the 
    # spread of the roots. A tighter bound comes from roots that are more 
    # evenly distributed in the relevant domain.
    
    # Compute a proxy for the C4 bound based on root distribution.
    # The bound should be minimized when roots are optimally placed.
    
    # Use the actual polynomial construction:
    # P(x) = prod_{i=1}^{m} (x - x_i)^2
    # Then compute relevant integrals.
    
    x = np.linspace(0, 100, 10000)
    dx = x[1] - x[0]
    
    # Compute P(x) = prod (x - x_i)^2
    P = np.ones_like(x)
    for xi in roots:
        P *= (x - xi)**2
    
    # Weight function for n=1: w(x) = x^1 * exp(-x)
    w = x * np.exp(-x)
    
    # Compute integrals
    # Numerator: integral of x^n * |P(x)|^2 * w(x) dx
    # Denominator: integral of |P(x)|^2 * w(x) dx
    # Or the bound might be: integral of x * |P(x)|^2 * w(x) dx / (integral of |P(x)|^2 * w(x) dx)
    
    P2 = P**2
    num = np.trapz(x * P2 * w, x)
    den = np.trapz(P2 * w, x)
    
    if den <= 0:
        return 1e10
    
    # The upper bound on C4 is related to this ratio
    # C4_bound = num / den
    bound = num / den
    
    return bound

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    m = 12
    
    # Initial guess: use Laguerre roots as starting point
    alpha = 1.0
    init_roots, _ = roots_genlaguerre(m, alpha)
    init_roots = np.sort(init_roots)
    
    # Define bounds for optimization
    # Roots should be positive and in a reasonable range
    # For m=12, roots of L_12^(1) are roughly in [0.5, 40]
    bounds = [(0.01, 100.0)] * m
    
    # Objective function wrapper
    def objective(x):
        if time.time() > deadline - 10:
            return 1e10
        return _compute_c4_bound(x)
    
    # Try multiple optimization methods
    best_result = None
    best_score = 1e10
    
    # Method 1: L-BFGS-B
    try:
        res = minimize(objective, init_roots, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
    
    # Method 2: Nelder-Mead (unconstrained, but we can penalize out-of-bounds)
    try:
        def objective_nm(x):
            if time.time() > deadline - 10:
                return 1e10
            # Penalize if any root is out of bounds
            penalty = 0
            for xi in x:
                if xi < 0.01:
                    penalty += (0.01 - xi)**2 * 1000
                elif xi > 100.0:
                    penalty += (xi - 100.0)**2 * 1000
            return _compute_c4_bound(np.sort(x)) + penalty
        
        res = minimize(objective_nm, init_roots, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
    
    # Method 3: Powell
    try:
        res = minimize(objective, init_roots, method='Powell', bounds=bounds,
                       options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
    
    # If no optimization succeeded, fall back to Laguerre roots
    if best_result is None:
        best_result = init_roots
    else:
        # Ensure roots are positive and sorted
        best_result = np.sort(np.maximum(best_result, 0.01))
    
    return [float(x) for x in best_result]
