"""Refined Cohn-Goncalves Laguerre construction: multi-restart 12-DOF optimization with Laguerre-node seeds."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
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
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _laguerre_node_seeds(m=12, n_roots=12):
    """
    Generate diverse initial root configurations based on generalized Laguerre nodes.
    These are natural candidates because the weight x^alpha * exp(-x) matches the
    orthogonality measure. We vary alpha and scaling to get diverse starting points.
    """
    seeds = []
    
    # Standard Laguerre nodes (alpha=0)
    nodes, _ = roots_laguerre(n_roots)
    seeds.append(nodes.copy())
    
    # Generalized Laguerre with various alpha values
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        try:
            from scipy.special import roots_genlaguerre
            nodes, _ = roots_genlaguerre(n_roots, alpha)
            if np.all(nodes > 0):
                seeds.append(nodes.copy())
        except Exception:
            pass
    
    # Scaled versions of standard nodes
    for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
        seeds.append(nodes * scale)
    
    # Log-spaced roots
    for base in [0.3, 0.5, 0.7, 1.0]:
        r = np.geomspace(base, base * 100, n_roots)
        seeds.append(r)
    
    # Linear spaced
    for start, end in [(0.5, 50), (1, 100), (2, 200), (0.1, 30)]:
        seeds.append(np.linspace(start, end, n_roots))
    
    # Filter: all positive and finite
    valid_seeds = []
    for s in seeds:
        if np.all(s > 0) and np.all(np.isfinite(s)):
            valid_seeds.append(s)
    
    return valid_seeds


def construct():
    m = 12
    n_roots = 12
    
    # Generate diverse initial guesses from Laguerre nodes and other families
    all_seeds = _laguerre_node_seeds(m, n_roots)
    
    # Include a simple linear seed as fallback
    linear_seed = np.array([4.0 * (i + 1) for i in range(n_roots)])
    all_seeds.append(linear_seed)
    
    # Evaluate all seeds to find the best starting point
    best_roots = linear_seed.copy()
    best_val = _c4_bound(best_roots, m)
    
    for seed in all_seeds:
        if _time_left() < 30:
            break
        val = _c4_bound(seed, m)
        if val < best_val:
            best_val = val
            best_roots = seed.copy()
    
    # Phase 1: Multi-restart L-BFGS-B in log-space (12 DOF)
    # Work in log-space to enforce positivity naturally
    def obj_log(log_roots):
        roots = np.exp(log_roots)
        if np.any(~np.isfinite(roots)):
            return 1e10
        return _c4_bound(roots, m)
    
    log_bounds = [(-2.0, 6.0)] * n_roots  # roots in [~0.13, ~403]
    
    # Collect starting points in log-space
    log_starts = [np.log(best_roots)]
    
    for seed in all_seeds:
        if np.all(seed > 0.01):
            log_starts.append(np.log(seed))
    
    # Add perturbed versions of the best
    for scale in [0.5, 0.8, 1.2, 1.5]:
        perturbed = best_roots * scale
        if np.all(perturbed > 0.01) and np.all(perturbed < 400):
            log_starts.append(np.log(perturbed))
    
    # Shuffle for diversity
    rng = np.random.RandomState(42)
    rng.shuffle(log_starts)
    
    max_restarts = min(len(log_starts), 40)
    for i in range(max_restarts):
        if _time_left() < 20:
            break
        x0 = log_starts[i]
        if np.any(x0 < -2.0) or np.any(x0 > 6.0):
            x0 = np.clip(x0, -1.9, 5.9)
        try:
            res = minimize(
                obj_log,
                x0=x0,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res.success or res.fun < best_val:
                roots = np.exp(res.x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    val = _c4_bound(roots, m)
                    if val < best_val:
                        best_val = val
                        best_roots = roots.copy()
        except Exception:
            continue
    
    # Phase 2: Nelder-Mead derivative-free refinement from current best
    if _time_left() > 30:
        def obj_nm(roots_arr):
            if np.any(roots_arr <= 0):
                return 1e10
            return _c4_bound(roots_arr, m)
        
        x0 = best_roots.copy()
        try:
            res = minimize(
                obj_nm,
                x0=x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res.fun < best_val and np.all(res.x > 0):
                best_val = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Another round of L-BFGS from the Nelder-Mead result
    if _time_left() > 20:
        x0 = np.log(best_roots)
        x0 = np.clip(x0, -1.9, 5.9)
        try:
            res = minimize(
                obj_log,
                x0=x0,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res.fun < best_val:
                roots = np.exp(res.x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    val = _c4_bound(roots, m)
                    if val < best_val:
                        best_val = val
                        best_roots = roots.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    if _time_left() > 15:
        def obj_nm2(roots_arr):
            if np.any(roots_arr <= 0):
                return 1e10
            return _c4_bound(roots_arr, m)
        
        try:
            res = minimize(
                obj_nm2,
                x0=best_roots.copy(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if res.fun < best_val and np.all(res.x > 0):
                best_val = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass
    
    return [float(r) for r in best_roots]
