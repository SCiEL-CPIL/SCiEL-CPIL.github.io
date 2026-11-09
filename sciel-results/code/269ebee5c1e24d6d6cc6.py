"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize
from scipy.integrate import quad

M = 12  # number of double roots
N = 1   # dimension parameter

def _compute_bound(roots):
    """
    Compute the upper bound C4 for the given set of positive double roots.
    The bound is 1/||P||^2 where P(x) = prod (x - r_i)^2 with weight x^alpha * e^{-x}.
    Smaller values are better.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != M:
        return np.inf
    if np.any(roots <= 0):
        return np.inf
    if not np.all(np.diff(np.sort(roots)) > 1e-15):
        return np.inf
    
    roots_sorted = np.sort(roots)
    alpha = 2 * N - 1  # alpha = 1
    
    def integrand(x):
        if x < 0:
            return 0.0
        # Calculate P(x) = prod (x - r_i)^2
        # To avoid overflow/underflow in intermediate steps, we can compute log P(x)
        # but for x in [0, inf) and roots ~ O(10), P(x) can be large.
        # However, P(x)^2 * x * e^{-x} will decay.
        # Let's compute directly first.
        p_val = 1.0
        for r in roots_sorted:
            diff = x - r
            p_val *= diff * diff
        return p_val * p_val * x * math.exp(-x)
        
    try:
        # Use quad for numerical integration
        # Limit the upper bound to a reasonable value since e^{-x} decays fast
        # Roots are typically around 1..20. P(x)^2 x e^{-x} is negligible for x > 100
        norm_sq, err = quad(integrand, 0, 200, limit=100)
    except Exception:
        return np.inf

    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / norm_sq

def construct():
    """Find optimal 12 positive double-root locations."""
    deadline = time.time() + 570  # Safety margin before 600s limit
    
    best_roots = None
    best_bound = np.inf
    
    def update_best(roots, bound):
        nonlocal best_roots, best_bound
        if np.isfinite(bound) and bound < best_bound:
            best_bound = bound
            best_roots = np.sort(roots.copy())
    
    # Approach 1: Zeros of Laguerre polynomial L_m^{(alpha)}(x)
    try:
        n_zeros = M
        alpha_val = 2 * N - 1  # = 1
        
        # Construct the Jacobi matrix for L_m^{(alpha)}
        # The eigenvalues of the Jacobi matrix are the roots of L_m^{(alpha)}
        d = np.array([2*i + alpha_val - 1 for i in range(n_zeros)])
        e = np.array([math.sqrt((i+1)*(i+1+alpha_val)) for i in range(n_zeros-1)])
        
        J = np.zeros((n_zeros, n_zeros))
        for i in range(n_zeros):
            J[i, i] = d[i]
        for i in range(n_zeros-1):
            J[i, i+1] = e[i]
            J[i+1, i] = e[i]
        
        zeros = np.sort(np.linalg.eigvalsh(J))
        if np.all(zeros > 0) and len(zeros) == M:
            bound_laguerre = _compute_bound(zeros)
            update_best(zeros, bound_laguerre)
    except Exception:
        pass
    
    # Approach 2: Local polish from best found using L-BFGS-B
    if best_roots is not None and time.time() < deadline - 60:
        try:
            def local_objective(x):
                if np.any(x <= 0):
                    return 1e100 # Penalty
                if not np.all(np.diff(np.sort(x)) > 1e-12):
                    return 1e100
                return _compute_bound(x)
            
            # We need bounds for L-BFGS-B. 
            # Roots of L_12^1 are approx in [0.5, 20]. 
            # Let's use wide bounds.
            bounds = [(1e-6, 100.0)] * M
            
            # Use L-BFGS-B
            result = minimize(
                local_objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < np.inf:
                polished_roots = np.sort(result.x)
                if np.all(polished_roots > 0) and np.all(np.diff(polished_roots) > 1e-12):
                    update_best(polished_roots, result.fun)
        except Exception:
            pass
            
    # Approach 3: Try a few random perturbations and polish
    if time.time() < deadline - 30:
        np.random.seed(42)
        base_roots = best_roots if best_roots is not None else np.array([4.0 * (i + 1) for i in range(12)])
        
        for i in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb roots slightly
            perturbed = base_roots + np.random.normal(0, 0.1, M)
            # Ensure positive and sorted
            perturbed = np.sort(np.abs(perturbed) + 1e-6)
            
            try:
                def obj_pert(x):
                    if np.any(x <= 0):
                        return 1e100
                    if not np.all(np.diff(np.sort(x)) > 1e-12):
                        return 1e100
                    return _compute_bound(x)
                
                res = minimize(
                    obj_pert,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 100.0)] * M,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    polished_roots = np.sort(res.x)
                    if np.all(polished_roots > 0) and np.all(np.diff(polished_roots) > 1e-12):
                        update_best(polished_roots, res.fun)
            except Exception:
                pass

    # Return best found or fallback
    if best_roots is not None:
        return [float(r) for r in best_roots]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
