import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
from scipy.integrate import quad
import time

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses log-sum-exp for numerical stability.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12 or np.any(roots <= 0) or np.any(np.diff(roots) < 1e-8):
        return 1e100

    def integrand(x):
        if x < 0:
            return 0.0
        
        # Calculate log(P(x)^2) = 4 * sum(log|x - r_i|)
        # P(x) = prod (x - r_i)^2
        # P(x)^2 = prod (x - r_i)^4
        # log(P(x)^2) = 4 * sum(log|x - r_i|)
        
        # Handle x == r_i case (log(0) = -inf, exp(-inf)=0, but P(x)=0 so integrand is 0)
        # However, near roots, the function is small.
        
        terms = np.abs(x - roots)
        # Avoid log(0)
        terms = np.maximum(terms, 1e-300)
        
        log_p_sq = 4.0 * np.sum(np.log(terms))
        
        # Integrand is P(x)^2 * e^-x
        log_integrand = log_p_sq - x
        
        # Clamp to prevent overflow/underflow issues in exp
        if log_integrand > 700:
            return np.exp(700)
        if log_integrand < -700:
            return 0.0
            
        return np.exp(log_integrand)

    # Integrate from 0 to infinity
    # The polynomial grows as x^24, so e^-x decays faster.
    # We can integrate up to a large limit, e.g., 200, where e^-200 is negligible.
    try:
        val, err = quad(integrand, 0, 200, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val
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
            # Note: _compute_l2_norm_sq is expensive (uses quad).
            # We limit maxiter to prevent timeout.
            
            # Estimate time per iteration: quad takes ~0.1-1s.
            # 580s / 1s = 580 iterations max.
            # SLSQP is slow. Let's try 50 iterations first.
            
            result = minimize(
                objective,
                roots0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success and result.fun < best_score:
                best_roots = result.x
                best_score = result.fun
            
            # If time remains, do a second pass with tighter tolerance
            if _remaining() > 20.0:
                result2 = minimize(
                    objective,
                    best_roots,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
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
