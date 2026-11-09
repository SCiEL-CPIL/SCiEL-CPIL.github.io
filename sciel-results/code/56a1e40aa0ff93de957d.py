"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, gamma
from scipy.optimize import minimize, differential_evolution

M = 12  # number of double roots
N = 1   # dimension parameter

def _compute_bound_fast(roots):
    """
    Compute the upper bound C4 for the given set of positive double roots.
    Uses vectorized polynomial construction and gamma function for speed.
    Smaller values are better.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != M:
        return np.inf
    if np.any(roots <= 0):
        return np.inf
    if not np.all(np.diff(np.sort(roots)) > 1e-14):
        return np.inf
    
    roots_sorted = np.sort(roots)
    alpha = 2 * N - 1  # alpha = 1
    
    # Build P(x) = prod_{i=1}^{m} (x - r_i)^2 in monomial basis
    # P(x) = sum_{k=0}^{2m} c_k * x^k
    # Start with [1.0] and multiply by (x - r)^2 = x^2 - 2r*x + r^2 for each root
    coeffs = np.array([1.0])
    for r in roots_sorted:
        # Multiply by (x^2 - 2r*x + r^2)
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[2:] += coeffs
        new_coeffs[1:-1] += -2.0 * r * coeffs
        new_coeffs[0:-2] += r**2 * coeffs
        coeffs = new_coeffs
    
    # Compute ||P||^2 = int_0^inf P(x)^2 * x^alpha * e^{-x} dx
    # P(x)^2 has coefficients p2_coeffs where p2_coeffs[k] = sum_{i+j=k} c_i * c_j
    p2_coeffs = np.convolve(coeffs, coeffs)
    
    # ||P||^2 = sum_{k=0}^{4m} p2_coeffs[k] * gamma(k + alpha + 1)
    # Use log-space to avoid overflow for large k
    k_vals = np.arange(len(p2_coeffs))
    log_gamma_terms = np.array([math.lgamma(k + alpha + 1) for k in k_vals])
    
    # Compute in log-space to handle large values
    # norm_sq = sum(p2_coeffs[k] * gamma(k+alpha+1))
    # Some p2_coeffs may be negative, so we need to be careful
    # For now, try direct computation with scaling
    max_log = np.max(log_gamma_terms)
    scaled_terms = p2_coeffs * np.exp(log_gamma_terms - max_log)
    norm_sq = np.sum(scaled_terms) * np.exp(max_log)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / norm_sq

def _laguerre_zeros(m, alpha):
    """Get zeros of Laguerre polynomial L_m^{(alpha)} using Jacobi matrix eigenvalues."""
    d = np.array([2.0*i + alpha - 1.0 for i in range(m)])
    e = np.array([math.sqrt((i+1.0)*(i+1.0+alpha)) for i in range(m-1)])
    
    J = np.zeros((m, m))
    for i in range(m):
        J[i, i] = d[i]
    for i in range(m-1):
        J[i, i+1] = e[i]
        J[i+1, i] = e[i]
    
    zeros = np.sort(np.linalg.eigvalsh(J))
    return zeros

def construct():
    """Find optimal 12 positive double-root locations."""
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit
    
    best_roots = None
    best_bound = np.inf
    
    def update_best(roots, bound):
        nonlocal best_roots, best_bound
        if np.isfinite(bound) and bound < best_bound:
            best_bound = bound
            best_roots = np.sort(roots.copy())
    
    # Approach 1: Zeros of Laguerre polynomial L_m^{(alpha)}(x) as initial guess
    try:
        alpha_val = 2 * N - 1  # = 1
        zeros = _laguerre_zeros(M, alpha_val)
        if np.all(zeros > 0) and len(zeros) == M:
            bound_laguerre = _compute_bound_fast(zeros)
            update_best(zeros, bound_laguerre)
    except Exception:
        pass
    
    # Approach 2: Differential evolution with log-parameterized roots
    try:
        def objective_log(log_roots):
            if time.time() > deadline - 90:
                return np.inf
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(roots > 1000):
                return np.inf
            if not np.all(np.diff(np.sort(roots)) > 1e-10):
                return np.inf
            return _compute_bound_fast(roots)
        
        # Initial guess in log-space from Laguerre zeros
        alpha_val = 2 * N - 1
        try:
            lag_zeros = _laguerre_zeros(M, alpha_val)
            log_initial = np.log(lag_zeros)
        except:
            log_initial = np.log(np.linspace(0.5, 50, M))
        
        bounds_log = [(math.log(0.01), math.log(200.0))] * M
        
        result = differential_evolution(
            objective_log, bounds_log,
            maxiter=40,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.fun < np.inf:
            candidate_roots = np.sort(np.exp(result.x))
            if np.all(candidate_roots > 0) and np.all(np.diff(candidate_roots) > 1e-10):
                update_best(candidate_roots, result.fun)
    except Exception:
        pass
    
    # Approach 3: Local polish with Nelder-Mead and L-BFGS-B
    if best_roots is not None and time.time() < deadline - 60:
        try:
            # Nelder-Mead polish (derivative-free, robust)
            def nm_objective(roots):
                if np.any(roots <= 0):
                    return np.inf
                if not np.all(np.diff(np.sort(roots)) > 1e-12):
                    return np.inf
                return _compute_bound_fast(roots)
            
            result_nm = minimize(
                nm_objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result_nm.fun < np.inf:
                polished = np.sort(result_nm.x)
                if np.all(polished > 0) and np.all(np.diff(polished) > 1e-12):
                    update_best(polished, result_nm.fun)
        except Exception:
            pass
        
        # L-BFGS-B polish with log-transform
        if time.time() < deadline - 40:
            try:
                def log_objective(log_roots):
                    roots = np.exp(log_roots)
                    if np.any(roots <= 0) or np.any(roots > 1000):
                        return np.inf
                    if not np.all(np.diff(np.sort(roots)) > 1e-12):
                        return np.inf
                    return _compute_bound_fast(roots)
                
                log_x0 = np.log(best_roots)
                bounds_log = [(math.log(0.001), math.log(500.0))] * M
                
                result_lbfgs = minimize(
                    log_objective,
                    log_x0,
                    method='L-BFGS-B',
                    bounds=bounds_log,
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                
                if result_lbfgs.fun < np.inf:
                    polished = np.sort(np.exp(result_lbfgs.x))
                    if np.all(polished > 0) and np.all(np.diff(polished) > 1e-12):
                        update_best(polished, result_lbfgs.fun)
            except Exception:
                pass
    
    # Approach 4: Final high-precision polish with very tight tolerances
    if best_roots is not None and time.time() < deadline - 20:
        try:
            def final_objective(roots):
                if np.any(roots <= 0):
                    return np.inf
                if not np.all(np.diff(np.sort(roots)) > 1e-15):
                    return np.inf
                return _compute_bound_fast(roots)
            
            result_final = minimize(
                final_objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result_final.fun < np.inf:
                polished = np.sort(result_final.x)
                if np.all(polished > 0) and np.all(np.diff(polished) > 1e-15):
                    update_best(polished, result_final.fun)
        except Exception:
            pass
    
    # Return best found or fallback
    if best_roots is not None:
        return [float(r) for r in best_roots]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
