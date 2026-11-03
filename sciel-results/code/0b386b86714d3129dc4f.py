"""Refined Cohn-Goncalves Laguerre construction: geometric-root + direct optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_DEADLINE = 600.0
_START = time.time()

def _time_left():
    return _DEADLINE - time.time()

def _laguerre_coeffs(n, alpha=1):
    """
    Compute coefficients of L_n^(alpha)(x) in ascending power of x.
    """
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = (math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff

def _compute_norm_sq(poly):
    """
    Compute ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    fact = np.ones(max_k + 2)
    for j in range(1, max_k + 2):
        fact[j] = fact[j - 1] * j
    weights = fact[1:max_k + 2]
    norm_sq = np.dot(sq, weights)
    return norm_sq

def _c4_bound(roots, m=12):
    """
    Compute upper bound on C4 for given roots.
    Returns the bound value (lower is better).
    """
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq

def _norm_and_grad(roots, m=12):
    """
    Compute norm_sq and its gradient with respect to roots.
    roots: array of 12 positive distinct values.
    """
    coeff_L = _laguerre_coeffs(m, 1)
    
    max_deg = 50
    fact = np.ones(max_deg + 10)
    for j in range(1, max_deg + 10):
        fact[j] = fact[j - 1] * j
    
    P = coeff_L.copy()
    for r in roots:
        q = np.array([r * r, -2.0 * r, 1.0])
        P = np.convolve(P, q)
        
    P_sq = np.convolve(P, P)
    norm_sq = np.dot(P_sq, fact[1:len(P_sq)+1])
    
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 0.0, np.zeros(len(roots))
    
    grad = np.zeros(len(roots))
    
    for i, r_i in enumerate(roots):
        R = coeff_L.copy()
        for j, r_j in enumerate(roots):
            if j == i:
                continue
            q = np.array([r_j * r_j, -2.0 * r_j, 1.0])
            R = np.convolve(R, q)
            
        R_sq = np.convolve(R, R)
        
        moments = R_sq * (fact[2:len(R_sq)+2] - r_i * fact[1:len(R_sq)+1])
        I = np.sum(moments)
        
        grad[i] = -4.0 * I
        
    return norm_sq, grad

def _neg_log_norm_and_grad(roots, m=12):
    """
    Objective and gradient for minimization.
    """
    if _time_left() < 2:
        return 1e100, np.zeros(len(roots))
        
    if np.any(roots <= 0):
        return 1e100, np.zeros(len(roots))
        
    norm_sq, grad_N = _norm_and_grad(roots, m)
    
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e100, np.zeros(len(roots))
        
    obj = -math.log(norm_sq)
    grad_obj = -grad_N / norm_sq
    
    return obj, grad_obj

def _geometric_roots(log_a, log_b, n_roots=12):
    """Generate geometric roots: r_i = a * b^i for i=0..n_roots-1"""
    a = math.exp(log_a)
    b = math.exp(log_b)
    if b <= 1.001 or a <= 0:
        return None
    roots = [a * (b ** i) for i in range(n_roots)]
    if any(r <= 0 for r in roots):
        return None
    if max(roots) > 1e6:
        return None
    return roots

def _objective_geometric(params, m=12):
    """Objective for geometric parameter optimization."""
    log_a, log_b = params
    roots = _geometric_roots(log_a, log_b)
    if roots is None:
        return 1e100
    val = _c4_bound(roots, m)
    if not np.isfinite(val):
        return 1e100
    return val

def _nelder_mead_minimize(func, x0, bounds, maxiter=500, time_budget=60):
    """Nelder-Mead with time guard."""
    start_time = time.time()
    n = len(x0)
    
    def obj(x):
        if time.time() - start_time > time_budget:
            return 1e100
        try:
            v = func(x)
            return v if np.isfinite(v) else 1e100
        except Exception:
            return 1e100
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if np.isfinite(result.fun):
            return result.x, result.fun
    except Exception:
        pass
    return x0, func(x0)

def construct():
    m = 12
    n_roots = 12
    
    # Baseline: the seed roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0
        
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Optimize geometric parameters (a, b) via Nelder-Mead
    # r_i = a * b^i, optimize in log-space: (log_a, log_b)
    # b must be > 1, so log_b > 0
    
    # Initial guesses: various (log_a, log_b) pairs
    initial_guesses = [
        [math.log(1.0), math.log(1.5)],
        [math.log(2.0), math.log(1.3)],
        [math.log(0.5), math.log(2.0)],
        [math.log(5.0), math.log(1.2)],
        [math.log(1.0), math.log(1.8)],
        [math.log(3.0), math.log(1.4)],
        [math.log(0.8), math.log(1.6)],
        [math.log(2.0), math.log(1.1)],
        [math.log(1.5), math.log(1.7)],
        [math.log(4.0), math.log(1.35)],
    ]
    
    # Coarse scan over geometric parameters to find good basins
    coarse_candidates = []
    log_a_vals = np.linspace(math.log(0.1), math.log(20.0), 15)
    log_b_vals = np.linspace(math.log(1.05), math.log(3.0), 20)
    
    for la in log_a_vals:
        if _time_left() < 300:
            break
        for lb in log_b_vals:
            if _time_left() < 300:
                break
            roots = _geometric_roots(la, lb)
            if roots is None:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val):
                coarse_candidates.append((val, la, lb, roots[:]))
    
    coarse_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates + initial guesses for Nelder-Mead
    nm_starts = []
    for val, la, lb, roots in coarse_candidates[:8]:
        nm_starts.append((la, lb))
    for guess in initial_guesses:
        nm_starts.append(guess)
    
    # Deduplicate
    seen = set()
    unique_starts = []
    for s in nm_starts:
        key = (round(s[0], 4), round(s[1], 4))
        if key not in seen:
            seen.add(key)
            unique_starts.append(s)
    
    # Phase 2: Nelder-Mead on geometric parameters from multiple starts
    for start in unique_starts:
        if _time_left() < 120:
            break
            
        x0 = np.array(start)
        # Constrain: log_a in [-3, 5], log_b in [0.01, 2]
        x0 = np.clip(x0, [-3.0, 0.01], [5.0, 2.0])
        
        x_opt, f_opt = _nelder_mead_minimize(_objective_geometric, x0, None, maxiter=300, time_budget=30)
        
        if np.isfinite(f_opt) and f_opt < best_val:
            roots = _geometric_roots(x_opt[0], x_opt[1])
            if roots is not None and all(r > 0 for r in roots):
                cand_val = _c4_bound(roots, m)
                if np.isfinite(cand_val) and cand_val < best_val:
                    best_val = cand_val
                    best_roots = roots[:]
    
    # Phase 3: Local refinement with L-BFGS-B on the 12 individual roots
    if _time_left() > 90:
        x0 = np.array(best_roots)
        
        def fun_grad(x):
            if np.any(x <= 0):
                return 1e100, np.zeros_like(x)
            return _neg_log_norm_and_grad(x, m)
            
        try:
            result = minimize(
                fun_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(1e-3, 1000.0)] * n_roots,
                options={'maxiter': 1500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = result.x
                if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                    cand_val = _c4_bound(candidate_roots, m)
                    if np.isfinite(cand_val) and cand_val < best_val:
                        best_val = cand_val
                        best_roots = candidate_roots.tolist()
                        
        except Exception:
            pass
    
    # Phase 4: Final polish
    if _time_left() > 45:
        x0 = np.array(best_roots)
        
        def fun_grad(x):
            if np.any(x <= 0):
                return 1e100, np.zeros_like(x)
            return _neg_log_norm_and_grad(x, m)
            
        try:
            result = minimize(
                fun_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(1e-3, 1000.0)] * n_roots,
                options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = result.x
                if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                    cand_val = _c4_bound(candidate_roots, m)
                    if np.isfinite(cand_val) and cand_val < best_val:
                        best_val = cand_val
                        best_roots = candidate_roots.tolist()
                        
        except Exception:
            pass
    
    # Ensure we return valid positive distinct roots
    if len(best_roots) != n_roots or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
        
    return best_roots
# EVOLVE-BLOCK-END
