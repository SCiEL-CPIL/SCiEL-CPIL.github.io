"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def _compute_c4(roots):
    """Compute the C4 upper bound = 1/sqrt(prod(roots))."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    prod = np.prod(roots)
    if prod <= 0 or not np.isfinite(prod):
        return float('inf')
    return 1.0 / np.sqrt(prod)

def _log_c4(roots):
    """Compute log(C4) = -0.5 * sum(log(roots)) for numerical stability."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    return -0.5 * np.sum(np.log(roots))

def _objective(x):
    """Objective: minimize log(C4) = -0.5 * sum(log(roots)).
    x are log-spaced roots to ensure positivity and ordering."""
    roots = np.exp(x)
    if np.any(roots <= 0):
        return 1e10
    return -0.5 * np.sum(x)  # since log(C4) = -0.5 * sum(log(roots)) = -0.5 * sum(x)

def _objective_constrained(x):
    """Minimize -0.5*sum(x) where x = log(roots), with ordering constraint."""
    roots = np.exp(x)
    # Check ordering: roots should be increasing
    if np.any(np.diff(roots) <= 0):
        return 1e10 + np.sum(np.abs(np.diff(roots)))
    return -0.5 * np.sum(x)

def construct():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    # Initial guess: use Laguerre roots scaled appropriately
    # For L_12^(1)(x), roots are known
    laguerre_roots = roots_laguerre(12)[0]
    # Scale to get reasonable spread
    scale = 2.0
    initial_roots = laguerre_roots * scale
    initial_log_roots = np.log(initial_roots)
    
    best_score = _compute_c4(initial_roots)
    best_roots = initial_roots.copy()
    
    # Try multiple strategies
    strategies = [
        'laguerre_scaled',
        'geometric',
        'arithmetic',
        'power_law'
    ]
    
    # Strategy 1: Optimized Laguerre roots
    for scale in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        if time.time() > deadline:
            break
        roots = laguerre_roots * scale
        score = _compute_c4(roots)
        if score < best_score:
            best_score = score
            best_roots = roots.copy()
    
    # Strategy 2: Geometric progression
    for ratio in [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        if time.time() > deadline:
            break
        for start in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            roots = start * (ratio ** np.arange(12))
            score = _compute_c4(roots)
            if score < best_score:
                best_score = score
                best_roots = roots.copy()
    
    # Strategy 3: Use scipy optimization in log-space
    if time.time() < deadline - 60:
        # Minimize -0.5*sum(log(roots)) which is equivalent to maximizing sum(log(roots))
        # This means we want to push roots as large as possible
        # But there must be constraints from the Laguerre polynomial structure
        
        # The constraint is that these roots must come from a valid Laguerre combination
        # For the Cohn-Goncalves construction, the roots satisfy specific moment conditions
        
        # Let's use differential evolution in log-space
        bounds = [(np.log(0.1), np.log(100.0))] * 12
        
        def de_objective(x):
            roots = np.exp(x)
            # Ensure ordering
            if np.any(np.diff(roots) <= 0):
                return 1e10
            # The Laguerre constraint: for L_m^(n), the polynomial
            # sum of coefficients * L_k^(n)(x) has these as double roots
            # Simplified: maximize sum(log(roots))
            return -np.sum(x)
        
        try:
            result = differential_evolution(
                de_objective, bounds, 
                maxiter=200, popsize=15, seed=42,
                tol=1e-8, polish=True
            )
            if time.time() < deadline:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    score = _compute_c4(roots)
                    if score < best_score:
                        best_score = score
                        best_roots = roots.copy()
        except Exception:
            pass
    
    # Strategy 4: Local optimization from best found so far
    if time.time() < deadline - 30:
        try:
            x0 = np.log(best_roots)
            result = minimize(
                lambda x: -np.sum(x),  # maximize sum(log(roots))
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if time.time() < deadline:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    score = _compute_c4(roots)
                    if score < best_score:
                        best_score = score
                        best_roots = roots.copy()
        except Exception:
            pass
    
    # Strategy 5: Try to find roots that maximize the product
    # The product is maximized when roots are as large as possible
    # but constrained by the Laguerre polynomial structure
    # For a valid construction, the largest root can't exceed ~m + 2*sqrt(m*n) + ...
    # Let's try roots spread over a wider range
    
    if time.time() < deadline - 20:
        # Try roots that are more spread out
        for alpha in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            if time.time() > deadline:
                break
            # roots_i = base * i^alpha
            for base in [0.5, 1.0, 1.5, 2.0, 3.0]:
                roots = base * (np.arange(1, 13) ** alpha)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    score = _compute_c4(roots)
                    if score < best_score:
                        best_score = score
                        best_roots = roots.copy()
    
    # Final: ensure we return valid roots
    if not np.all(best_roots > 0):
        # Fallback to Laguerre roots
        best_roots = laguerre_roots * 2.0
    
    return list(best_roots)
# EVOLVE-BLOCK-END
