"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses multi-start with generalized Laguerre nodes + DE global search + high-precision L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
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
    
    # Gradient computation
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        R_i = np.ones(1)
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]]))
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
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

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Initial guesses: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)
    
    # Try several generalized Laguerre parameters as initial guesses
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0) and np.isfinite(xg).all():
                c4g = _compute_c4(xg)
                if c4g < best_c4:
                    best_c4 = c4g
                    best_roots = xg.copy()
        except Exception:
            pass

    # Try geometric spacing patterns
    for base in [1.2, 1.5, 2.0, 3.0]:
        try:
            geom = np.sort(np.array([base**i for i in range(12)]))
            c4g = _compute_c4(geom)
            if np.isfinite(c4g) and c4g < best_c4:
                best_c4 = c4g
                best_roots = geom.copy()
        except Exception:
            pass

    # Differential evolution for global search - multiple passes with different seeds
    for de_seed in [42, 137, 256, 999]:
        if not time_ok():
            break
        bounds = [(1e-4, 200.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=de_seed,
                maxiter=200,
                popsize=25,
                tol=1e-12,
                polish=False,
                mutation=(0.4, 1.2),
                recombination=0.8,
                update_interval=5
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # L-BFGS-B refinement from best DE point and several others
    candidates_to_refine = [best_roots.copy()]
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for _ in range(3):
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            candidates_to_refine.append(pert)
    
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
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final high-precision polish: multiple passes with tightening tolerances
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 30}
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
