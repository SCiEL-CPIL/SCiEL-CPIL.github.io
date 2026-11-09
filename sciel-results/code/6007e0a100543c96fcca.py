"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
from scipy.integrate import quad

def _objective(params):
    """
    Compute the uncertainty bound objective: the L^2 norm of P(x) = prod(x - r_i)^2
    with Laguerre weight e^{-x}. We minimize this to tighten the bound.
    """
    roots = np.sort(params)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e10  # penalty for invalid roots
    
    def integrand(x):
        # P(x) = prod(x - r_i)^2
        prod_val = 1.0
        for r in roots:
            prod_val *= (x - r) ** 2
        return prod_val ** 2 * np.exp(-x)
    
    integral, _ = quad(integrand, 0, np.inf, limit=200)
    return integral

def _objective_neg(params):
    """
    Negative log of the integral, to maximize the integral (if that's the right direction).
    """
    roots = np.sort(params)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e10
    
    def integrand(x):
        prod_val = 1.0
        for r in roots:
            prod_val *= (x - r) ** 2
        return prod_val ** 2 * np.exp(-x)
    
    integral, _ = quad(integrand, 0, np.inf, limit=200)
    if integral <= 0:
        return 1e10
    return -np.log(integral)

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses constrained optimization to minimize the Cohn-Goncalves uncertainty bound
    by treating the 12 double-root locations as continuous variables.
    """
    # Get initial guesses from Laguerre zeros
    zeros, _ = roots_laguerre(12)
    x0 = np.array(zeros)
    
    # Define bounds: roots must be positive and ordered
    bounds = [(1e-6, None)] * 12
    
    # Try multiple initializations based on asymptotic distribution
    best_result = None
    best_obj = np.inf
    
    # Multi-start: perturb the Laguerre zeros
    for trial in range(5):
        if trial == 0:
            x0_trial = x0.copy()
        else:
            # Perturb with decreasing magnitude
            scale = 0.1 / (trial + 1)
            x0_trial = x0 + scale * np.random.randn(12)
            x0_trial = np.sort(x0_trial)
            x0_trial = np.maximum(x0_trial, 1e-6)
        
        result = minimize(
            _objective,
            x0_trial,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_result = result
    
    if best_result is not None and best_result.fun < 1e10:
        roots_opt = np.sort(best_result.x)
        return [float(r) for r in roots_opt]
    else:
        # Fallback to Laguerre zeros
        return [float(z) for z in zeros]
# EVOLVE-BLOCK-END
