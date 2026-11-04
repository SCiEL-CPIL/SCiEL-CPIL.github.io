"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Strategy: 1) 1D search over generalized Laguerre alpha parameter, 2) multi-start DE, 3) high-precision L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
# We need k! for k up to 48 (degree of Q = P^2 where P is degree 24)
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf).
    coeffs[k] is coefficient of x^k. Result = sum_k coeffs[k] * k!."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    # Use dot product; factorial grows fast so for large degree this can overflow in float64
    # but we normalize by I1^2 so it should be OK if we scale
    return float(np.dot(coeffs[:min(n, 60)], _FACTS[:min(n, 60)]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs).
    Coefficients in ascending power order (np.polynomial.polynomial convention)."""
    # P(x) = prod (x - x_i)^2 = [prod (x - x_i)]^2
    # Build base polynomial B(x) = prod (x - x_i), then P = B^2
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # ascending order
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
    
    # Gradient: dC4/dx_i = (dI2/dx_i * I1^2 - I2 * 2 * I1 * dI1/dx_i) / I1^4
    # = dI2/dx_i / I1^2 - 2 * I2 * dI1/dx_i / I1^3
    # dI1/dx_i: dP/dx_i where P = (x-x_i)^2 * R_i, R_i = prod_{j!=i}(x-x_j)^2
    # dP/dx_i = -2(x-x_i) * R_i
    # So dI1/dx_i = -2 * integral((x-x_i)*R_i * e^{-x} dx)
    # Similarly P2 = P^2, dP2/dx_i = 2*P*dP/dx_i = -4*(x-x_i)*R_i*P
    # dI2/dx_i = -4 * integral((x-x_i)*R_i*P * e^{-x} dx)
    
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        R_i = np.ones(1)
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]]))
        # dI1: -2 * int (x-x_i)*R_i e^{-x} dx
        # (x-x_i)*R_i = x*R_i - x_i*R_i
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))  # multiply by x
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        # dI2: -4 * int (x-x_i)*R_i*P e^{-x} dx
        RP = np.polynomial.polynomial.polymul(R_i, P)
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

def _c4_from_alpha(alpha):
    """Compute C4 for roots that are generalized Laguerre nodes with parameter alpha."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if np.any(xg <= 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0  # return 45s before kill
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: 1D search over alpha for generalized Laguerre nodes
    best_roots = None
    best_c4 = np.inf
    
    # Coarse grid search over alpha
    alphas_coarse = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    for alpha in alphas_coarse:
        c4a = _c4_from_alpha(alpha)
        if c4a < best_c4:
            best_c4 = c4a
            # Get the roots for this alpha
            xg, _ = roots_genlaguerre(12, alpha)
            best_roots = np.sort(xg)
    
    # Fine 1D optimization using minimize_scalar
    if time_ok():
        try:
            # First find a good bracket
            alpha_best = 1.0  # default
            for alpha in alphas_coarse:
                c4a = _c4_from_alpha(alpha)
                if c4a < best_c4:
                    best_c4 = c4a
                    alpha_best = alpha
            
            # Optimize in a neighborhood of the best coarse alpha
            bracket = (max(0.01, alpha_best * 0.5), alpha_best * 2.0)
            res_1d = minimize_scalar(
                _c4_from_alpha,
                bounds=bracket,
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 200}
            )
            if res_1d.success and np.isfinite(res_1d.fun) and res_1d.fun < best_c4:
                best_c4 = res_1d.fun
                xg, _ = roots_genlaguerre(12, res_1d.x)
                best_roots = np.sort(xg)
        except Exception:
            pass
    
    # Fallback if 1D search failed
    if best_roots is None or not np.isfinite(best_c4):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Differential evolution for global search - two passes with different seeds
    for de_seed in [42, 137]:
        if not time_ok():
            break
        bounds = [(1e-4, 150.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=de_seed,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False,
                mutation=(0.3, 1.0),
                recombination=0.8,
                update_interval=10
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B refinement from best point and several others
    candidates_to_refine = [best_roots.copy()]
    # Add perturbed versions with varying scales
    rng = np.random.default_rng(123)
    for scale in [0.05, 0.1, 0.2]:
        for _ in range(3):
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            candidates_to_refine.append(pert)
    
    # Also add generalized Laguerre nodes as refinement starting points
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0):
                candidates_to_refine.append(xg.copy())
        except Exception:
            pass

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * 12,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision polish with multiple passes
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 20}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break  # No improvement, stop polishing
        except Exception:
            break

    # Final safety check and return
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        # Fallback to standard Laguerre roots if something went wrong
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
