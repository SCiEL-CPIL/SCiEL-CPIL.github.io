import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Pre-compute Gauss-Laguerre nodes and weights for high accuracy
# Degree of P(x)^2 is 24. N=20 is exact for polynomials up to degree 39.
_GL_N = 20
try:
    _gl_nodes, _gl_weights = np.polynomial.laguerre.laggauss(_GL_N)
except Exception:
    _gl_nodes = np.linspace(0, 100, _GL_N)
    _gl_weights = np.ones(_GL_N) / _GL_N

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses Gauss-Laguerre quadrature for speed.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
    
    try:
        diff = _gl_nodes[:, np.newaxis] - roots[np.newaxis, :]
        abs_diff = np.abs(diff)
        # Avoid log(0)
        abs_diff[abs_diff == 0] = 1e-300
        log_p_sq = 4.0 * np.sum(np.log(abs_diff), axis=1)
        
        log_w = np.log(_gl_weights)
        log_terms_w = log_p_sq + log_w
        
        max_log = np.max(log_terms_w)
        if np.isnan(max_log):
             return 1e100
             
        if max_log > 700:
             return 1e100
             
        sum_exp = np.sum(np.exp(log_terms_w - max_log))
        result = np.exp(max_log) * sum_exp
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
    
    best_roots = roots0.copy()
    best_score = _compute_l2_norm_sq(best_roots)
    
    # Optimize in log-space to ensure positivity
    # y = log(r) -> r = exp(y)
    # Objective: f(y) = _compute_l2_norm_sq(exp(y))
    
    def objective_log(y):
        if _remaining() < 5.0:
            return best_score
        r = np.exp(y)
        return _compute_l2_norm_sq(r)
    
    y0 = np.log(roots0)
    
    # Bounds for y: r in [1e-6, 1e6] -> y in [-13.8, 13.8]
    bounds_y = [(-15.0, 15.0) for _ in range(12)]
    
    try:
        if _remaining() > 10.0:
            # L-BFGS-B is efficient for box constraints
            result = minimize(
                objective_log,
                y0,
                method='L-BFGS-B',
                bounds=bounds_y,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.fun < best_score:
                best_roots = np.exp(result.x)
                best_score = result.fun
            
            # Second pass with tighter tolerance if time allows
            if _remaining() > 20.0:
                result2 = minimize(
                    objective_log,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds_y,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
                )
                if result2.fun < best_score:
                    best_roots = np.exp(result2.x)
                    best_score = result2.fun
                    
    except Exception:
        pass
    
    # Final validation
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.diff(best_roots) < 1e-8):
        # Fallback to initial guess if optimization produced invalid roots
        best_roots = roots0
    
    return [float(r) for r in best_roots]
