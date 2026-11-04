"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction optimizer."""
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    log_P = np.zeros(n_quad)
    for r in roots:
        diff = x_quad - r
        abs_diff = np.abs(diff)
        log_P += 2.0 * np.log(abs_diff + 1e-300)
    
    max_log_P = np.max(log_P)
    if max_log_P > 700:
        return np.inf
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2) or np.isnan(int_P) or np.isnan(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    if np.isnan(c4) or np.isinf(c4) or c4 < 0:
        return np.inf
    return c4

def _compute_c4_fast(roots):
    """Faster C4 computation using direct polynomial evaluation."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2 = [prod (x - r_i)]^2
    # Use log for stability
    log_prod = np.zeros(n_quad)
    for r in roots:
        log_prod += np.log(np.abs(x_quad - r) + 1e-300)
    
    log_P = 2.0 * log_prod
    max_lp = np.max(log_P)
    if max_lp > 700:
        return np.inf
    
    P = np.exp(log_P)
    
    int_P = np.dot(w_quad, P)
    int_P2 = np.dot(w_quad, P * P)
    
    if int_P <= 0 or not np.isfinite(int_P) or not np.isfinite(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4 if np.isfinite(c4) and c4 > 0 else np.inf

def _roots_from_power_sums(powers):
    """
    Given first 12 power sums p_k = sum_i r_i^k for k=1..12,
    recover the 12 roots using Newton's identities to get elementary
    symmetric polynomials, then find roots of the monic polynomial.
    """
    n = 12
    # p[0] = sum r_i^0 = n = 12
    # p[1] = sum r_i
    # p[2] = sum r_i^2
    # ...
    # p[12] = sum r_i^12
    
    # Newton's identities: e_k = (1/k) * sum_{j=1}^{k} (-1)^{j-1} e_{k-j} * p_j
    # where e_0 = 1
    e = np.zeros(n + 1)
    e[0] = 1.0
    
    for k in range(1, n + 1):
        s = 0.0
        for j in range(1, k + 1):
            s += ((-1) ** (j - 1)) * e[k - j] * powers[j - 1]
        e[k] = s / k
    
    # The monic polynomial is: x^n - e_1*x^(n-1) + e_2*x^(n-2) - ... + (-1)^n e_n
    # Coefficients for np.roots (highest degree first)
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1.0
    for k in range(1, n + 1):
        coeffs[k] = ((-1) ** k) * e[k]
    
    # Find roots
    try:
        all_roots = np.roots(coeffs)
    except Exception:
        return None
    
    # Filter for real positive roots
    real_roots = []
    for r in all_roots:
        if abs(r.imag) < 1e-8 and r.real > 1e-10:
            real_roots.append(r.real)
    
    if len(real_roots) < n:
        return None
    
    # We need exactly 12 positive real roots
    # Take the 12 largest (or all if exactly 12)
    real_roots.sort(reverse=True)
    if len(real_roots) >= n:
        return np.sort(np.array(real_roots[:n]))
    
    return None

def _objective_from_powers(powers):
    """Objective: compute C4 from power sums."""
    if np.any(powers < -1e-10) or np.any(np.isnan(powers)) or np.any(np.isinf(powers)):
        return 1e10
    
    roots = _roots_from_power_sums(powers)
    if roots is None:
        return 1e10
    
    return _compute_c4_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 570
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # Strategy 1: Direct optimization in log-space with multiple starts
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [x0.copy()]
    
    # Geometric sequences
    for ratio in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
        geo_roots = np.array([1.0 * (ratio ** i) for i in range(12)])
        if np.all(geo_roots > 0) and np.max(geo_roots) < 100:
            initial_points.append(geo_roots)
    
    # Perturbed versions
    for _ in range(15):
        perturbed = x0 * (1 + 0.4 * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Log-space optimization
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
        
        try:
            log_init = np.log(x_init)
            result_log = minimize(
                lambda lr: _compute_c4_fast(np.exp(lr)),
                log_init,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if np.isfinite(result_log.fun) and result_log.fun < best_score:
                candidate = np.exp(result_log.x)
                candidate = np.sort(candidate)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 30:
        return [float(xi) for xi in best_roots]
    
    # Strategy 2: Power-sum parameterization
    # Compute power sums from current best
    p_best = np.array([np.sum(best_roots ** k) for k in range(1, 13)])
    
    # Optimize in power-sum space
    try:
        def _ps_objective(ps):
            if np.any(ps < -1e-8):
                return 1e10
            roots = _roots_from_power_sums(ps)
            if roots is None:
                return 1e10
            return _compute_c4_fast(roots)
        
        # Multi-start in power-sum space
        for _ in range(5):
            ps_init = p_best * (1 + 0.1 * rng.standard_normal(12))
            ps_init = np.maximum(ps_init, 1e-6)
            
            result_ps = minimize(
                _ps_objective,
                ps_init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            if np.isfinite(result_ps.fun) and result_ps.fun < best_score:
                roots_ps = _roots_from_power_sums(result_ps.x)
                if roots_ps is not None:
                    candidate = np.sort(roots_ps)
                    if np.all(candidate > 0):
                        score = _compute_c4_fast(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return [float(xi) for xi in best_roots]
    
    # Strategy 3: Differential Evolution
    try:
        de_bounds = [(1e-3, 50.0)] * 12
        result_de = differential_evolution(
            _compute_c4_fast,
            bounds=de_bounds,
            seed=42,
            maxiter=80,
            tol=1e-12,
            polish=True,
            workers=1,
            popsize=15
        )
        if np.isfinite(result_de.fun) and result_de.fun < best_score:
            candidate = np.sort(result_de.x)
            if np.all(candidate > 0):
                score = _compute_c4_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return [float(xi) for xi in best_roots]
    
    # Final polish
    try:
        log_best = np.log(best_roots)
        result_final = minimize(
            lambda lr: _compute_c4_fast(np.exp(lr)),
            log_best,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if np.isfinite(result_final.fun) and result_final.fun < best_score:
            candidate = np.exp(result_final.x)
            candidate = np.sort(candidate)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                score = _compute_c4_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    except Exception:
        pass
    
    return [float(xi) for xi in best_roots]
