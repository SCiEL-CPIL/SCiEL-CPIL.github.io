import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Precompute factorials up to 24!
_MAX_DEG = 24
_FACTS = np.array([math.factorial(k) for k in range(_MAX_DEG + 1)], dtype=float)

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses analytic expansion: integral = sum b_k * k!
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
        
    try:
        # Coefficients of Q(x) = prod (x - r_i)
        # np.poly returns [a_n, ..., a_0]
        coeffs_q = np.poly(roots)
        
        # Coefficients of P(x) = Q(x)^2
        # Convolution of coeffs_q with itself
        coeffs_p = np.convolve(coeffs_q, coeffs_q)
        
        # coeffs_p is [b_24, ..., b_0]
        # We want sum b_k * k!
        # _FACTS is [0!, 1!, ..., 24!]
        # So we need to reverse _FACTS to align with coeffs_p
        result = np.dot(coeffs_p, _FACTS[::-1])
        
        return result
    except Exception:
        return 1e100

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    m = 12
    
    # Initial guess: roots of generalized Laguerre polynomial L_12^(0)(x)
    try:
        roots0 = roots_genlaguerre(m, 0)
        roots0 = np.sort(roots0.real)
        if len(roots0) != 12 or np.any(roots0 <= 0):
            roots0 = np.array([4.0 * (i + 1) for i in range(12)])
    except Exception:
        roots0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Calculate baseline score
    base_score = _compute_l2_norm_sq(roots0)
    
    best_roots = roots0.copy()
    best_score = base_score
    
    # Define objective for optimizer
    def objective(x):
        return _compute_l2_norm_sq(x)
    
    # Bounds: each root must be positive and ordered
    bounds = [(1e-6, None) for _ in range(12)]
    
    # Constraints: roots must be sorted (r_i < r_{i+1})
    constraints = []
    for i in range(11):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 1e-6})
    
    try:
        # Attempt optimization if time permits
        if _remaining() > 10.0:
            # Use SLSQP for constrained optimization
            # Objective is now analytic and fast, so we can run many iterations.
            
            result = minimize(
                objective,
                roots0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success and result.fun < best_score:
                best_roots = result.x
                best_score = result.fun
            
            # If time remains, do a second pass with tighter tolerance
            if _remaining() > 10.0:
                result2 = minimize(
                    objective,
                    best_roots,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
                )
                if result2.success and result2.fun < best_score:
                    best_roots = result2.x
                    best_score = result2.fun
                    
    except Exception:
        pass
    
    # Ensure final result is valid
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.diff(best_roots) < 1e-8):
        best_roots = roots0
    
    return [float(r) for r in best_roots]
