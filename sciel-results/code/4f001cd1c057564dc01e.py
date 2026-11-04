"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Combined strategy: 1D alpha search + multi-start DE + high-precision L-BFGS-B polish.
Key improvements: efficient gradient via log-derivatives, better initial guesses, more robust polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_p_coeffs(roots):
    """Build P(x) = prod (x-x_i)^2 in ascending coeff order. Returns P_coeffs (degree 24)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # degree 12, ascending
    P = polymul(B, B)          # degree 24
    return P

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 where I1=int P e^{-x}, I2=int P^2 e^{-x}, P=prod(x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P = _build_p_coeffs(roots)
    I1 = _poly_moment(P)
    if I1 <= 0 or not np.isfinite(I1):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # P^2 = P * P
    from numpy.polynomial.polynomial import polymul
    P2 = polymul(P, P)
    I2 = _poly_moment(P2)
    if not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Gradient computation using log-derivative approach for efficiency
    # dP/dx_i = P * d(log P)/dx_i = P * (-2 * sum_j 1/(x-x_j) for j!=i)
    # Actually d/dx_i [prod (x-x_j)^2] = P * d/dx_i [log prod (x-x_j)^2]
    # = P * sum_j d/dx_i [2 log(x-x_j)]
    # For j=i: d/dx_i [2 log(x-x_i)] = 2/(x-x_i) -- but this is the singular part
    # For j!=i: d/dx_i [2 log(x-x_j)] = 0 (x_j doesn't depend on x_i)
    # Wait, that's wrong. x_i appears in the factor (x-x_i)^2.
    # P = (x-x_i)^2 * R_i where R_i = prod_{j!=i}(x-x_j)^2
    # dP/dx_i = -2(x-x_i) * R_i = -2(x-x_i) * P / (x-x_i)^2 = -2 P / (x-x_i)
    # So dP/dx_i = -2P/(x-x_i)
    # dI1/dx_i = int dP/dx_i * e^{-x} dx = -2 * int P/(x-x_i) * e^{-x} dx
    # Similarly P^2 = P^2, d(P^2)/dx_i = 2P * dP/dx_i = -4P^2/(x-x_i)
    # dI2/dx_i = -4 * int P^2/(x-x_i) * e^{-x} dx

    # We need to compute int P/(x-x_i) e^{-x} dx and int P^2/(x-x_i) e^{-x} dx
    # P/(x-x_i) = prod_{j!=i}(x-x_j)^2 * (x-x_i) -- this is a polynomial!
    # P^2/(x-x_i) = P * P/(x-x_i) = P * prod_{j!=i}(x-x_j)^2*(x-x_i) -- also polynomial

    grad = np.zeros(n)
    inv_I1 = 1.0 / I1
    inv_I1_sq = inv_I1 * inv_I1
    inv_I1_cu = inv_I1_sq * inv_I1

    for i in range(n):
        # Compute S_i(x) = P(x)/(x-x_i) = prod_{j!=i}(x-x_j)^2 * (x-x_i)
        # This has degree 23
        # Actually P/(x-x_i) = [prod_{j!=i}(x-x_j)^2] * (x-x_i)
        # Let's build it as: B_without_i = prod_{j!=i}(x-x_j), then S_i = B_without_i^2 * (x-x_i)
        # But that's degree 22+1=23. Alternatively, just do polynomial division.
        
        # More efficient: build R_i = prod_{j!=i}(x-x_j)^2 (degree 22), then S_i = R_i * (x-x_i) (degree 23)
        from numpy.polynomial.polynomial import polyfromroots, polymul, polydiv
        
        # Use polynomial division: P / (x - x_i)
        # P is degree 24, dividing by (x-x_i) gives degree 23 quotient + remainder
        # Since x_i is a double root, the remainder should be 0
        divisor = np.array([-roots[i], 1.0])  # x - x_i in ascending order
        S_i, _ = polydiv(P, divisor)  # S_i = P/(x-x_i), degree 23
        
        # dI1/dx_i = -2 * int S_i * e^{-x} dx
        dI1_i = -2.0 * _poly_moment(S_i)
        
        # dI2/dx_i = -4 * int P * S_i * e^{-x} dx
        PS_i = polymul(P, S_i)  # degree 24+23=47
        dI2_i = -4.0 * _poly_moment(PS_i)
        
        grad[i] = dI2_i * inv_I1_sq - 2.0 * I2 * dI1_i * inv_I1_cu

    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for generalized Laguerre roots with parameter alpha."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if np.all(xg > 0) and np.all(np.isfinite(xg)):
            return _compute_c4(xg)
    except Exception:
        pass
    return np.inf

def construct():
    start = time.time()
    DEADLINE = 600.0
    SAFETY = 30.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    # Phase 1: 1D search over generalized Laguerre parameter alpha
    # This gives a good initial guess based on root density profile
    best_roots = None
    best_c4 = np.inf

    # Grid search over alpha
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        if not time_ok():
            break
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if np.all(xg > 0) and np.all(np.isfinite(xg)):
            c4g = _compute_c4(xg)
            if c4g < best_c4:
                best_c4 = c4g
                best_roots = xg.copy()

    # Standard Laguerre as fallback
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Differential evolution for global search
    # Use the best alpha-derived roots as a starting point in the population
    if time_ok():
        bounds = [(1e-4, 200.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=250,
                popsize=18,
                tol=1e-10,
                polish=False,
                mutation=(0.4, 1.0),
                recombination=0.7,
                update_interval=8
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Second DE run with different seed if time permits
    if time_ok():
        try:
            de_result2 = differential_evolution(
                _objective,
                bounds=bounds,
                seed=137,
                maxiter=150,
                popsize=15,
                tol=1e-10,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.6,
                update_interval=10
            )
            if np.all(de_result2.x > 0) and np.isfinite(de_result2.fun):
                candidate = np.sort(de_result2.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B refinement from multiple starting points
    candidates_to_refine = [best_roots.copy()]
    
    # Perturbed versions
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.1, 0.2, 0.35]:
        for _ in range(2):
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            candidates_to_refine.append(pert)

    # Add best alpha-derived roots
    for alpha in [0.5, 1.0, 2.0]:
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
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish passes
    for polish_iter in range(4):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
