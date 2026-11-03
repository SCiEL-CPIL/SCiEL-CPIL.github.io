"""Refined Cohn-Goncalves Laguerre polynomial construction optimizer.
Combines structured parameterization (geometric/arithmetic) for fast global search
with high-precision gradient-based local refinement.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

# Precompute factorials for moment computation
# P has degree 24, P^2 has degree 48. We need k! for k up to 48.
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf).
    coeffs[k] is coefficient of x^k. Result = sum_k coeffs[k] * k!."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    # Limit to precomputed factorials
    limit = min(n, 60)
    return float(np.dot(coeffs[:limit], _FACTS[:limit]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs).
    Coefficients in ascending power order (np.polynomial.polynomial convention)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # ascending order, degree 11
    P = polymul(B, B)         # degree 22? No.
    # Wait. P(x) = prod (x-x_i)^2.
    # Let B(x) = prod (x-x_i). Degree 11.
    # P(x) = B(x)^2. Degree 22.
    # P2(x) = P(x)^2 = B(x)^4. Degree 44.
    # The task says "12 double roots".
    # If P has double roots, P(x) = \prod (x-x_i)^2.
    # Degree is 12*2 = 24.
    # My previous thought: B = prod(x-x_i) has degree 12.
    # Let's re-verify.
    # roots = [x_1, ..., x_12].
    # polyfromroots(roots) gives \prod (x-x_i). Degree 12.
    # P = B^2. Degree 24.
    # P2 = P^2. Degree 48.
    # So we need factorials up to 48.
    
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Gradient calculation
    # C4 = I2 / I1^2
    # dC4/dx_i = (dI2/dx_i * I1^2 - I2 * 2 * I1 * dI1/dx_i) / I1^4
    #           = dI2/dx_i / I1^2 - 2 * I2 * dI1/dx_i / I1^3
    
    grad = np.zeros_like(roots)
    n_roots = len(roots)
    
    # Pre-compute P for reuse in dI2
    # dP/dx_i = -2(x-x_i) R_i, where R_i = prod_{j!=i} (x-x_j)^2
    # dI1/dx_i = -2 * int (x-x_i) R_i e^{-x} dx
    # dI2/dx_i = 2 * int P * dP/dx_i e^{-x} dx = -4 * int (x-x_i) R_i P e^{-x} dx
    
    for i in range(n_roots):
        # Build R_i = prod_{j!=i} (x-x_j)^2
        # This is P / (x-x_i)^2
        # To avoid division instability, build from scratch or use synthetic division?
        # Building from scratch is O(N^2) per i, total O(N^3). N=12, so 1728 ops. Cheap.
        
        R_i_coeffs = np.ones(1)
        for j in range(n_roots):
            if j == i:
                continue
            # Multiply by (x - x_j)^2 = x^2 - 2x_j x + x_j^2
            # In ascending order: [x_j^2, -2x_j, 1]
            quad = np.array([roots[j]**2, -2*roots[j], 1.0])
            R_i_coeffs = np.polynomial.polynomial.polymul(R_i_coeffs, quad)
        
        # term1 = int (x-x_i) R_i e^{-x} dx
        # (x-x_i) R_i = x R_i - x_i R_i
        xR_i = np.polynomial.polynomial.polymul(R_i_coeffs, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i_coeffs)
        dI1_i = -2.0 * term1
        
        # term2 = int (x-x_i) R_i P e^{-x} dx
        # RP = R_i * P
        RP = np.polynomial.polynomial.polymul(R_i_coeffs, P)
        xRP = np.polynomial.polynomial.polymul(RP, np.array([0.0, 1.0]))
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 30.0  # return 30s before kill
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    best_roots = None
    best_c4 = np.inf

    def update_best(candidate_roots, c4_val):
        nonlocal best_roots, best_c4
        if c4_val < best_c4 and np.all(candidate_roots > 0) and np.all(np.isfinite(candidate_roots)):
            best_roots = np.sort(candidate_roots)
            best_c4 = c4_val

    # 1. Initial Guess: Standard Laguerre Roots
    # These are the roots of the Laguerre polynomial L_12.
    # They provide a good baseline structure.
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_init = _compute_c4(x0)
    update_best(x0, c4_init)

    # 2. Structured Search: Geometric Progression
    # x_i = a * b^i. Parameters: log_a, log_b.
    # This reduces dimensionality from 12 to 2.
    def _geo_obj(params):
        log_a, log_b = params
        # Ensure b is close to 1 or > 1 to keep roots sorted and positive
        # If b <= 1, roots decrease. We sort later, but optimization might struggle.
        # Let's constrain b > 1.0
        if log_b < np.log(1.01):
            return 1e10
        a = np.exp(log_a)
        b = np.exp(log_b)
        roots = a * (b ** np.arange(12))
        if np.any(roots <= 0) or np.any(roots > 1e6):
            return 1e10
        return _compute_c4(roots)

    if time_ok():
        # Multistart for Geometric
        rng = np.random.default_rng(42)
        # Heuristic: Laguerre roots are roughly linear, so b is close to 1.
        # Mean ratio of consecutive Laguerre roots
        ratios = x0[1:] / x0[:-1]
        mean_log_b = np.mean(np.log(ratios))
        
        for _ in range(30):
            init_log_b = mean_log_b + rng.normal(0, 0.2)
            init_log_a = np.log(x0[0]) + rng.normal(0, 0.2)
            init_params = [init_log_a, init_log_b]
            try:
                res = minimize(_geo_obj, init_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_c4:
                    log_a, log_b = res.x
                    a = np.exp(log_a)
                    b = np.exp(log_b)
                    candidate = a * (b ** np.arange(12))
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    update_best(candidate, c4)
            except Exception:
                pass

    # 3. Structured Search: Arithmetic Progression
    # x_i = a + i * d. Parameters: a, d.
    def _arith_obj(params):
        a, d = params
        if d <= 0: return 1e10
        roots = a + np.arange(12) * d
        if np.any(roots <= 0) or np.any(roots > 1e6):
            return 1e10
        return _compute_c4(roots)

    if time_ok():
        # Multistart for Arithmetic
        rng = np.random.default_rng(43)
        # Estimate d from Laguerre roots
        d_est = np.mean(np.diff(x0))
        a_est = x0[0]
        
        for _ in range(30):
            init_a = a_est + rng.normal(0, 0.2)
            init_d = d_est + rng.normal(0, 0.2)
            if init_d <= 0: init_d = 0.1
            if init_a <= 0: init_a = 0.1
            init_params = [init_a, init_d]
            try:
                res = minimize(_arith_obj, init_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_c4:
                    a, d = res.x
                    candidate = a + np.arange(12) * d
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    update_best(candidate, c4)
            except Exception:
                pass

    # 4. High-Precision Local Refinement
    # Use the best candidate found so far as the starting point for a full 12-dim optimization.
    # Also try refining the Laguerre roots directly.
    candidates_to_refine = [best_roots.copy(), x0.copy()]
    
    # Add some perturbed versions of the best structured roots
    rng = np.random.default_rng(123)
    for _ in range(5):
        pert = best_roots * (1.0 + 0.1 * rng.standard_normal(12))
        pert = np.maximum(pert, 1e-3)
        pert = np.sort(pert)
        candidates_to_refine.append(pert)

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            # Use L-BFGS-B with analytic gradient
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, 1000.0)] * 12,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    update_best(candidate, c4)
        except Exception:
            pass

    # 5. Final Polish
    # One more very tight refinement on the absolute best
    if time_ok() and best_roots is not None:
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, 1000.0)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    update_best(candidate, c4)
        except Exception:
            pass

    if best_roots is None:
        # Fallback to Laguerre if everything failed
        return [float(xi) for xi in x0]

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
