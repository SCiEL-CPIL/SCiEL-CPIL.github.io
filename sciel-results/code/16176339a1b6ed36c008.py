import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, roots_genlaguerre, gamma
from scipy.integrate import quad

# EVOLVE-BLOCK-START

def _remaining(deadline):
    """Return remaining time in seconds."""
    return max(0.0, deadline - time.time())

def _log_poly_sq_weighted_integral(roots):
    """
    Compute the logarithm of the weighted L2 norm squared of P(x)^2.
    P(x) = prod (x - r_i)^2.
    Integral = int_0^inf x * e^-x * P(x)^2 dx.
    Since P(x) = prod (x-r_i)^2, P(x)^2 = prod (x-r_i)^4.
    log(P(x)^2) = 4 * sum(log|x-r_i|).
    """
    roots = np.asarray(roots, dtype=float)
    
    def integrand(x):
        # Avoid log(0)
        diffs = np.abs(x - roots)
        diffs[diffs == 0] = 1e-300
        log_val = 4.0 * np.sum(np.log(diffs))
        # The term x * e^-x
        log_weight = np.log(x + 1e-300) - x
        # Total log value
        total_log = log_val + log_weight
        # Clip to prevent overflow in exp
        if total_log > 700:
            return np.exp(700)
        if total_log < -700:
            return 0.0
        return np.exp(total_log)
    
    # Integrate from 0 to infinity
    # Split the integral to help the quadrature
    try:
        val, err = quad(integrand, 0, np.inf, limit=100)
        if val <= 0:
            return 700.0 # Large value
        return math.log(val)
    except Exception:
        return 700.0

def _objective(roots):
    """
    Objective function to minimize.
    We assume the bound C4 is proportional to the weighted L2 norm of P(x)^2.
    Minimizing this norm minimizes the bound.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    # Ensure distinctness for stability
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-8):
        return 1e10
    
    return _log_poly_sq_weighted_integral(roots)

def _clamp_roots(roots):
    """Clamp roots to be positive and distinct."""
    roots = np.asarray(roots, dtype=float)
    roots = np.maximum(roots, 1e-6)
    roots = np.sort(roots)
    for i in range(1, len(roots)):
        if roots[i] - roots[i-1] < 1e-6:
            roots[i] = roots[i-1] + 1e-6
    return roots.tolist()

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    """
    deadline = time.time() + 590  # Safety margin
    
    # 1. Generate initial candidates
    candidates = []
    
    # Standard Laguerre roots L_12^(1)
    try:
        r1 = roots_genlaguerre(12, 1)['roots']
        candidates.append(sorted(r1.tolist()))
    except:
        pass
        
    # Laguerre roots L_12^(0)
    try:
        r0 = roots_genlaguerre(12, 0)['roots']
        candidates.append(sorted(r0.tolist()))
    except:
        pass
        
    # Laguerre roots L_12^(2)
    try:
        r2 = roots_genlaguerre(12, 2)['roots']
        candidates.append(sorted(r2.tolist()))
    except:
        pass
        
    # Scaled versions of L_12^(1)
    base_roots = candidates[0] if candidates else [4.0 * (i + 1) for i in range(12)]
    for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
        candidates.append(sorted([r * scale for r in base_roots]))
        
    # Evaluate all candidates
    best_roots = None
    best_score = float('inf')
    
    for cand in candidates:
        if _remaining(deadline) < 10:
            break
        try:
            score = _objective(cand)
            if score < best_score:
                best_score = score
                best_roots = cand.copy()
        except:
            pass
            
    # 2. Local Optimization
    if best_roots is not None and _remaining(deadline) > 60:
        try:
            # Use L-BFGS-B for local search
            res = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-3, 100.0)] * 12,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                cand_roots = _clamp_roots(res.x)
                cand_score = _objective(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots
        except:
            pass
            
    # 3. Differential Evolution (Global Search) if time permits
    if _remaining(deadline) > 120:
        try:
            bounds = [(0.1, 50.0)] * 12
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=20,
                tol=1e-8,
                polish=True
            )
            if de_res.success:
                cand_roots = _clamp_roots(de_res.x)
                cand_score = _objective(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots
        except:
            pass
            
    # 4. Final Polish
    if best_roots is not None and _remaining(deadline) > 10:
        try:
            res = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-3, 100.0)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.success:
                cand_roots = _clamp_roots(res.x)
                cand_score = _objective(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots
        except:
            pass
            
    # Fallback
    if best_roots is None or len(best_roots) != 12:
        try:
            best_roots = sorted(roots_genlaguerre(12, 1)['roots'].tolist())
        except:
            best_roots = [4.0 * (i + 1) for i in range(12)]
            
    return best_roots

# EVOLVE-BLOCK-END
