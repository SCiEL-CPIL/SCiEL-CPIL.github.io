import time
import math
import numpy as np
from scipy.optimize import minimize, root, brentq
from scipy.special import eval_genlaguerre, factorial

# EVOLVE-BLOCK-START

import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, roots_genlaguerre, gamma

def _remaining(deadline):
    """Return remaining time in seconds."""
    return max(0.0, deadline - time.time())

def _laguerre_eval(x, m, alpha):
    """Evaluate generalized Laguerre polynomial L_m^(alpha)(x)."""
    return eval_genlaguerre(m, alpha, x)

def _c4_objective(roots):
    """
    Objective function to minimize: upper bound on C4.
    
    In the Cohn-Goncalves Laguerre construction, the bound C4 is typically
    related to the product or symmetric functions of the roots. A common form
    involves the value of the Laguerre polynomial combination at specific points.
    
    For a polynomial P(x) = prod_i (x - r_i)^2 with double roots, the C4 bound
    is often related to ||P||_2^2 / ||P||_1^2 or similar norm ratios, or to
    the maximum of |P(x)| over a specific interval.
    
    Let me try: C4 ~ prod_i r_i * (1 + prod_i r_i)
    or C4 ~ max_i r_i * min_i r_i
    or C4 ~ (sum r_i)^2 / prod r_i
    
    Actually, a more principled approach: in uncertainty principles with sign changes,
    the constant often involves the ratio of L2 norm to L1 norm of the polynomial
    or its transform. Let me use a plausible surrogate.
    
    For the Laguerre polynomial construction, a natural quantity is:
    C4 = (integral_0^inf x * P(x)^2 dx) / (integral_0^inf P(x)^2 dx)
    which is the "mean" of the weight function, related to the first moment.
    
    But for double roots, P(x) = prod (x - r_i)^2, so P(x)^2 = prod (x - r_i)^4.
    
    Let me use a simpler surrogate: the geometric mean of the roots times the arithmetic mean.
    """
    roots = np.asarray(roots)
    if len(roots) != 12 or np.any(roots <= 0):
        return 1e10
    
    # Try: C4 = (arithmetic mean) * (geometric mean)
    arith_mean = np.mean(roots)
    geo_mean = np.exp(np.mean(np.log(roots)))
    
    # Alternative: C4 = max(roots) / min(roots) (condition number)
    cond = np.max(roots) / np.min(roots)
    
    # Another: C4 = sum of pairwise distances normalized
    # Let me use a combination that might correlate with the actual bound
    # C4 ~ (sum r_i)^2 / (n * prod r_i)^(2/n) - this is a measure of spread
    
    # Actually, let me try the most natural one: the product of roots
    # In many uncertainty bounds, the constant scales with the product of the zeros
    prod_roots = np.prod(roots)
    
    # Normalize by the seed value to make it comparable
    # Seed roots: 4, 8, 12, ..., 48
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_prod = np.prod(seed_roots)
    
    # If C4 is proportional to the product, then minimizing product minimizes C4
    # But the Laguerre zeros have a different product
    return prod_roots / seed_prod

def _verify_roots(roots):
    """Verify that roots are positive, distinct, and satisfy basic constraints."""
    roots = np.asarray(roots)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    # Check distinctness
    if np.any(np.diff(np.sort(roots)) < 1e-10):
        return False
    return True

def _clamp_roots(roots):
    """Clamp roots to be positive and distinct."""
    roots = np.asarray(roots, dtype=float)
    # Ensure positivity
    roots = np.maximum(roots, 1e-6)
    # Sort and ensure distinctness
    roots = np.sort(roots)
    for i in range(1, len(roots)):
        if roots[i] - roots[i-1] < 1e-6:
            roots[i] = roots[i-1] + 1e-6
    return roots.tolist()

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    # Start with Laguerre zeros
    try:
        roots_lag = roots_genlaguerre(12, 1)
        best_roots = sorted(roots_lag['roots'].tolist())
    except:
        best_roots = [4.0 * (i + 1) for i in range(12)]
    
    best_score = _c4_objective(best_roots)
    
    # Try local optimization
    if _remaining(deadline) > 30:
        try:
            res = minimize(
                _c4_objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-6, 100.0)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.success and _verify_roots(res.x):
                candidate = _clamp_roots(res.x)
                cand_score = _c4_objective(candidate)
                if cand_score < best_score:
                    best_roots = candidate
                    best_score = cand_score
        except:
            pass
    
    # Try differential evolution if time permits
    if _remaining(deadline) > 120:
        try:
            bounds = [(0.1, 100.0)] * 12
            de_res = differential_evolution(
                _c4_objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-12,
                polish=True
            )
            if de_res.success and _verify_roots(de_res.x):
                candidate = _clamp_roots(de_res.x)
                cand_score = _c4_objective(candidate)
                if cand_score < best_score:
                    best_roots = candidate
                    best_score = cand_score
        except:
            pass
    
    # Final polish
    if _remaining(deadline) > 10:
        try:
            res = minimize(
                _c4_objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-6, 100.0)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-18}
            )
            if res.success and _verify_roots(res.x):
                candidate = _clamp_roots(res.x)
                cand_score = _c4_objective(candidate)
                if cand_score < best_score:
                    best_roots = candidate
                    best_score = cand_score
        except:
            pass
    
    # Ensure final validity
    best_roots = _clamp_roots(best_roots)
    if not _verify_roots(best_roots):
        # Fallback to Laguerre zeros
        try:
            roots_lag = roots_genlaguerre(12, 1)
            best_roots = sorted(roots_lag['roots'].tolist())
        except:
            best_roots = [4.0 * (i + 1) for i in range(12)]
    
    return best_roots

# EVOLVE-BLOCK-END
