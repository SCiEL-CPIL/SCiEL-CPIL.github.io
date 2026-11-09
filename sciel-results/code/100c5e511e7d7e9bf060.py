"""
Rewritten program for the refined Cohn-Goncalves Laguerre polynomial task.
Uses the roots of generalized Laguerre polynomials as the primary structural candidate,
refined by local optimization.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0   # Safety margin before the 630 s wall-clock kill
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the uncertainty bound C4 for given double-root locations.
    C4 = [integral_0^inf x^4 e^{-x} |P(x)|^2 dx] / [integral_0^inf e^{-x} |P(x)|^2 dx]
    where P(x) = prod (x - r_i)^2.
    """
    r = np.asarray(r, dtype=float)
    
    # Use Gauss-Laguerre quadrature for integrals with e^{-x} weight
    # Degree of P^2 is 24. We need enough nodes for accuracy.
    # 50 nodes is usually sufficient for degree 24 polynomials.
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(50)
    
    # Evaluate P(x) at each node
    P_vals = np.ones_like(x_nodes)
    for ri in r:
        P_vals *= (x_nodes - ri) ** 2
    
    P2 = P_vals ** 2
    
    # Numerator: integral x^4 e^{-x} |P(x)|^2 dx
    num = np.sum(x_weights * x_nodes**4 * P2)
    # Denominator: integral e^{-x} |P(x)|^2 dx
    den = np.sum(x_weights * P2)
    
    if den < 1e-30:
        return 1e10  # Very large penalty for invalid roots
    
    return num / den

def _objective(r_flat):
    """
    Objective function for the optimizer.
    Ensures roots are positive and distinct.
    """
    if _deadline_reached():
        return 1e10
    
    r = np.asarray(r_flat, dtype=float)
    
    # Enforce positivity
    if np.any(r <= 0):
        return 1e10
        
    # Enforce distinctness (double roots must be at distinct locations)
    # Sort to check gaps
    r_sorted = np.sort(r)
    gaps = np.diff(r_sorted)
    if np.any(gaps < 1e-8):
        return 1e10
        
    return _compute_bound(r)

def _construct_core():
    """
    Generate candidate root sets and optimize.
    """
    candidates = []
    
    # Candidate 1: Roots of Generalized Laguerre Polynomial L_12^(1)(x)
    # This is the most structurally sound guess for n=1.
    try:
        nodes, _ = roots_genlaguerre(NROOTS, N)
        nodes = np.sort(nodes)
        candidates.append(nodes)
    except Exception:
        pass
        
    # Candidate 2: Roots of Standard Laguerre Polynomial L_12^(0)(x)
    # In case the weight function interpretation is different.
    try:
        nodes0, _ = roots_genlaguerre(NROOTS, 0)
        nodes0 = np.sort(nodes0)
        candidates.append(nodes0)
    except Exception:
        pass
        
    # Candidate 3: The Seed (Arithmetic Progression)
    seed_roots = np.array([4.0 * (i + 1) for i in range(NROOTS)])
    candidates.append(seed_roots)
    
    best_r = None
    best_score = float('inf')
    
    # Evaluate all candidates
    for cand in candidates:
        if _deadline_reached():
            break
        try:
            score = _compute_bound(cand)
            if score < best_score:
                best_score = score
                best_r = cand.copy()
        except Exception:
            continue
            
    if best_r is None:
        # Fallback to seed
        best_r = seed_roots
        best_score = _compute_bound(best_r)
        
    # Refine the best candidate using L-BFGS-B
    if not _deadline_reached():
        try:
            result = minimize(
                _objective,
                best_r,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_r = result.x.copy()
        except Exception:
            pass
            
    # Final check: ensure roots are positive and sorted
    best_r = np.sort(best_r)
    if np.any(best_r <= 0):
        # If optimization failed and went non-positive, revert to seed
        best_r = seed_roots
        
    return best_r.tolist()

# EVOLVE-BLOCK-END

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). Score is the upper bound on C4 (lower is better).
    """
    try:
        result = _construct_core()
    except Exception:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
        
    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
        
    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = [4.0 * (i + 1) for i in range(NROOTS)]
        
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
        
    return out
