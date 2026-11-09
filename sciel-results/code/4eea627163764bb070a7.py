"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.optimize import minimize, differential_evolution

M = 12  # number of double roots
N = 1   # dimension parameter

def _laguerre_k(x, k, alpha):
    """Generalized Laguerre polynomial L_k^{(alpha)}(x)."""
    return eval_genlaguerre(k, alpha, x)

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
    
    # Build P(x) = prod_{i=1}^{m} (x - r_i)^2 in monomial basis
    coeffs = np.array([1.0])
    for r in roots_sorted:
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[2:] += coeffs
        new_coeffs[1:-1] += -2*r * coeffs
        new_coeffs[0:-2] += r**2 * coeffs
        coeffs = new_coeffs
    
    # Compute ||P||^2 = int_0^inf P(x)^2 * x^alpha * e^{-x} dx
    p2_coeffs = np.convolve(coeffs, coeffs)
    
    norm_sq = 0.0
    for n in range(len(p2_coeffs)):
        norm_sq += p2_coeffs[n] * math.gamma(n + alpha + 1)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    # The upper bound on sign changes is proportional to 1/sqrt(norm_sq) or 1/norm_sq
    # For the Cohn-Gonçalves construction, we minimize 1/||P||^2
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
    
    # Approach 2: Differential evolution with multiple restarts
    try:
        def objective(x):
            if time.time() > deadline - 60:
                return np.inf
            if np.any(x <= 0):
                return np.inf
            if not np.all(np.diff(np.sort(x)) > 1e-10):
                return np.inf
            return _compute_bound(x)
        
        bounds = [(0.1, 100.0)] * M
        
        # Try multiple seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline - 120:
                break
            
            result = differential_evolution(
                objective, bounds,
                maxiter=30,
                popsize=12,
                tol=1e-8,
                seed=seed,
                polish=True
            )
            
            if result.success and result.fun < np.inf:
                best_roots_candidate = np.sort(result.x)
                if np.all(best_roots_candidate > 0) and np.all(np.diff(best_roots_candidate) > 1e-10):
                    update_best(best_roots_candidate, result.fun)
    except Exception:
        pass
    
    # Approach 3: Local polish from best found
    if best_roots is not None and time.time() < deadline - 30:
        try:
            def local_objective(x):
                if np.any(x <= 0):
                    return np.inf
                if not np.all(np.diff(np.sort(x)) > 1e-12):
                    return np.inf
                return _compute_bound(x)
            
            result = minimize(
                local_objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.fun < np.inf:
                polished_roots = np.sort(result.x)
                if np.all(polished_roots > 0) and np.all(np.diff(polished_roots) > 1e-12):
                    update_best(polished_roots, result.fun)
        except Exception:
            pass
    
    # Return best found or fallback
    if best_roots is not None:
        return [float(r) for r in best_roots]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
