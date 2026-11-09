import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # Safety margin before 630s limit

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
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-8):
        return 1e100

    def integrand(x):
        if x < 0:
            return 0.0
        
        terms = np.abs(x - roots)
        terms = np.maximum(terms, 1e-300)
        
        log_p_sq = 4.0 * np.sum(np.log(terms))
        log_integrand = log_p_sq - x
        
        if log_integrand > 700:
            return np.exp(700)
        if log_integrand < -700:
            return 0.0
            
        return np.exp(log_integrand)

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
    
    base_score = _compute_l2_norm_sq(roots0)
    best_roots = roots0.copy()
    best_score = base_score
    
    # Objective that enforces ordering via penalty
    def objective(x):
        x = np.sort(x)
        if np.any(x <= 0) or np.any(np.diff(x) < 1e-6):
            return 1e100
        return _compute_l2_norm_sq(x)
    
    # Bounds for each root
    bounds = [(1e-6, 100.0) for _ in range(12)]
    
    try:
        # Global optimization via differential evolution
        if _remaining() > 30.0:
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=False,
                updating='immediate',
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                best_roots = np.sort(result.x)
                best_score = result.fun
            
            # Local refinement with SLSQP if time permits
            if _remaining() > 30.0:
                def objective_local(x):
                    return _compute_l2_norm_sq(x)
                
                constraints = []
                for i in range(11):
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 1e-6})
                
                result2 = minimize(
                    objective_local,
                    best_roots,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
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
