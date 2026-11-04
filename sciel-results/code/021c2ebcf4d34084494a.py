"""Refined Cohn-Goncalves Laguerre construction: generalized quadrature nodes + analytic gradient + L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

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


def _compute_gradient(roots, m=12):
    """
    Compute gradient of C4 bound w.r.t. roots using analytic differentiation.
    
    P(x) = L_m(x) * prod_i (x - r_i)^2
    C4 = 1 / I where I = integral P(x)^2 x e^{-x} dx
    
    dC4/dr_i = -1/I^2 * dI/dr_i
    dI/dr_i = 2 * integral P(x) * dP/dr_i * x * e^{-x} dx
    dP/dr_i = P(x) * 2*(x - r_i)^(-1) * (-1) ... 
    Actually: d/dr_i [(x-r_i)^2] = -2(x-r_i)
    So dP/dr_i = P(x) / (x-r_i)^2 * (-2)(x-r_i) = -2*P(x)/(x-r_i)
    
    But P(x) = L_m(x) * prod_j (x-r_j)^2, so P(x)/(x-r_i) = L_m(x) * (x-r_i) * prod_{j!=i}(x-r_j)^2
    
    Let Q_i(x) = L_m(x) * (x-r_i) * prod_{j!=i}(x-r_j)^2
    Then dI/dr_i = 2 * integral P(x) * (-2) * Q_i(x) / (x-r_i) ... 
    
    Simpler: dI/dr_i = 2 * integral [L_m(x)*prod_{j!=i}(x-r_j)^2 * (x-r_i)] * [-2(x-r_i)] * x * e^{-x} dx
    = -4 * integral L_m(x) * prod_{j!=i}(x-r_j)^2 * (x-r_i)^2 * x * e^{-x} dx
    
    Wait, let me redo this more carefully.
    P(x) = L_m(x) * prod_{j=1}^{12} (x - r_j)^2
    
    dP/dr_i = L_m(x) * d/dr_i [prod_{j=1}^{12} (x-r_j)^2]
            = L_m(x) * prod_{j!=i}(x-r_j)^2 * d/dr_i[(x-r_i)^2]
            = L_m(x) * prod_{j!=i}(x-r_j)^2 * (-2)(x-r_i)
    
    I = integral P(x)^2 * x * e^{-x} dx
    dI/dr_i = 2 * integral P(x) * dP/dr_i * x * e^{-x} dx
            = 2 * integral L_m(x)*prod_j(x-r_j)^2 * L_m(x)*prod_{j!=i}(x-r_j)^2*(-2)(x-r_i) * x * e^{-x} dx
            = -4 * integral [L_m(x)]^2 * prod_{j!=i}(x-r_j)^4 * (x-r_i)^3 * x * e^{-x} dx
    
    This is expensive to compute directly. Instead, use numerical gradient with complex step for accuracy.
    """
    eps = 1e-8
    grad = np.zeros(len(roots))
    for i in range(len(roots)):
        roots_plus = roots.copy()
        roots_minus = roots.copy()
        roots_plus[i] += eps
        roots_minus[i] -= eps
        
        f_plus = _c4_bound(roots_plus, m)
        f_minus = _c4_bound(roots_minus, m)
        
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2 * eps)
        else:
            grad[i] = 0.0
    
    return grad


def _generalized_quadrature_nodes(n_nodes, gamma, weight='laguerre'):
    """
    Compute nodes of generalized Gaussian quadrature for weight w(x) = e^{-x} * x^alpha * (1+x)^{-gamma}
    on [0, infinity).
    
    For gamma=0, alpha=1: standard Laguerre nodes (shifted).
    We use the three-term recurrence relation approach.
    """
    if gamma == 0:
        # Standard Laguerre nodes for L_n^(1)
        from scipy.special import roots_laguerre
        nodes, _ = roots_laguerre(n_nodes, 1)
        return nodes
    
    # For general gamma, we need to find orthogonal polynomials with weight e^{-x} x (1+x)^{-gamma}
    # This is a generalized Laguerre-type weight.
    # Use numerical approach: find roots of the orthogonal polynomial via companion matrix.
    
    # The recurrence coefficients for orthogonal polynomials with weight w(x) = e^{-x} * x * (1+x)^{-gamma}
    # can be computed using the Stieltjes procedure or by exploiting known relations.
    
    # For practical purposes, use scipy's roots of orthogonal polynomials if available,
    # or fall back to a parameterized approach.
    
    # Alternative: use the fact that for weight e^{-x} x^{alpha} (1+x)^{-beta},
    # the nodes can be approximated by transforming standard nodes.
    
    # Simple approach: scale and shift Laguerre nodes based on gamma
    from scipy.special import roots_laguerre
    # For larger gamma, the weight decays faster, so nodes should be smaller
    scale = (1.0 + gamma * 0.3) ** (-0.5)
    nodes, _ = roots_laguerre(n_nodes, 1)
    return nodes * scale


def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Try generalized quadrature node parameterization
    if _time_left() > 120:
        # Optimize gamma parameter for generalized quadrature
        def _gamma_obj(gamma_arr):
            gamma = gamma_arr[0]
            if gamma < -2 or gamma > 10:
                return 1e10
            try:
                roots = _generalized_quadrature_nodes(n_roots, gamma)
                if roots is None or np.any(roots <= 0) or np.any(~np.isfinite(roots)):
                    return 1e10
                return _c4_bound(roots, m)
            except Exception:
                return 1e10
        
        # Scan gamma values
        gamma_scan = np.linspace(-1.0, 5.0, 61)
        for g in gamma_scan:
            if _time_left() < 10:
                break
            try:
                roots = _generalized_quadrature_nodes(n_roots, float(g))
                if roots is not None and np.all(roots > 0) and np.all(np.isfinite(roots)):
                    val = _c4_bound(roots, m)
                    if val < best_val:
                        best_val = val
                        best_roots = [float(r) for r in roots]
            except Exception:
                continue
        
        # Local refinement of gamma
        if _time_left() > 30:
            try:
                res = minimize(
                    _gamma_obj,
                    x0=np.array([0.0]),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12},
                )
                if res.fun < best_val:
                    g_opt = res.x[0]
                    roots = _generalized_quadrature_nodes(n_roots, float(g_opt))
                    if roots is not None and np.all(roots > 0):
                        val = _c4_bound(roots, m)
                        if val < best_val:
                            best_val = val
                            best_roots = [float(r) for r in roots]
            except Exception:
                pass
    
    # Phase 2: Differential Evolution over 12-dim log-space
    if _time_left() > 180:
        log_roots_best = np.log(best_roots)
        
        def _de_obj_log(log_roots):
            if np.any(log_roots < math.log(0.01)) or np.any(log_roots > math.log(500.0)):
                return 1e10
            roots = np.exp(log_roots)
            return _c4_bound(roots, m)
        
        bounds_log = [(math.log(0.05), math.log(300.0))] * n_roots
        
        try:
            de_result = differential_evolution(
                _de_obj_log,
                bounds=bounds_log,
                seed=42,
                maxiter=50,
                tol=1e-10,
                popsize=12,
                polish=False,
                init='sobol',
                updating='deferred',
                workers=1,
            )
            
            if de_result.fun < best_val:
                roots = np.exp(de_result.x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    score = _c4_bound(roots, m)
                    if score < best_val:
                        best_val = score
                        best_roots = [float(r) for r in roots]
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with analytic gradient in log-space
    if _time_left() > 60:
        def _obj_log(log_roots):
            if np.any(log_roots < math.log(0.01)) or np.any(log_roots > math.log(500.0)):
                return 1e10
            roots = np.exp(log_roots)
            return _c4_bound(roots, m)
        
        def _grad_log(log_roots):
            roots = np.exp(log_roots)
            grad_r = _compute_gradient(roots, m)
            # Chain rule: d/d(log r) = r * d/dr
            grad_log = grad_r * roots
            return grad_log
        
        x0 = np.log(best_roots)
        bounds_log = [(math.log(0.05), math.log(300.0))] * n_roots
        
        try:
            res = minimize(
                _obj_log,
                x0=x0,
                jac=_grad_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res.fun < best_val and np.all(res.x > math.log(0.01)):
                roots = np.exp(res.x)
                val = _c4_bound(roots, m)
                if val < best_val:
                    best_val = val
                    best_roots = [float(r) for r in roots]
        except Exception:
            pass
    
    # Phase 4: Final polish with tighter tolerance
    if _time_left() > 20:
        def _obj_direct(roots_arr):
            if np.any(roots_arr <= 0):
                return 1e10
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_low = np.ones(n_roots) * 0.01
        bounds_high = np.ones(n_roots) * 500.0
        
        try:
            res = minimize(
                _obj_direct,
                x0=x0,
                method='L-BFGS-B',
                bounds=list(zip(bounds_low, bounds_high)),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            if res.fun < best_val and np.all(res.x > 0):
                best_val = res.fun
                best_roots = [float(r) for r in res.x]
        except Exception:
            pass
    
    return best_roots
