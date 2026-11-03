"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses generalized Laguerre nodes (1D alpha search) + multi-start L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
# P has degree 24 (12 double roots), P^2 has degree 48
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

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # 1D search over alpha for generalized Laguerre nodes
    # roots_genlaguerre(n, alpha) gives nodes of L_n^(alpha)(x)
    # alpha = 0 is standard Laguerre
    best_roots = None
    best_c4 = np.inf

    # Scan alpha values around 0
    alpha_values = [0.0]
    # Add values on a log-spaced grid around 0, and some linear values
    for a in np.linspace(-5.0, 5.0, 51):
        alpha_values.append(a)
    # Also try some specific values
    alpha_values.extend([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 0.01, -0.01])

    alpha_values = sorted(set(alpha_values))

    for alpha in alpha_values:
        if not time_ok():
            break
        try:
            nodes, _ = roots_genlaguerre(12, alpha)
            nodes = np.sort(nodes)
            if np.any(nodes <= 0):
                continue
            c4 = _compute_c4(nodes)
            if np.isfinite(c4) and c4 < best_c4:
                best_c4 = c4
                best_roots = nodes.copy()
        except Exception:
            continue

    # If no valid roots found, fallback to standard Laguerre
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Local refinement from the best alpha point
    try:
        res = minimize(
            _objective_grad,
            best_roots.copy(),
            jac=True,
            method='L-BFGS-B',
            bounds=[(1e-8, None)] * 12,
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

    # Additional refinement from nearby alpha values
    if time_ok():
        # Find which alpha gave the best result and try neighbors
        # Re-scan a finer grid around the best alpha
        # First, find the best alpha from our scan
        best_alpha = 0.0
        best_alpha_c4 = np.inf
        for alpha in alpha_values:
            try:
                nodes, _ = roots_genlaguerre(12, alpha)
                nodes = np.sort(nodes)
                if np.any(nodes <= 0):
                    continue
                c4 = _compute_c4(nodes)
                if np.isfinite(c4) and c4 < best_alpha_c4:
                    best_alpha_c4 = c4
                    best_alpha = alpha
            except Exception:
                continue

        # Fine scan around best alpha
        fine_alphas = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 21)
        for alpha in fine_alphas:
            if not time_ok():
                break
            try:
                nodes, _ = roots_genlaguerre(12, alpha)
                nodes = np.sort(nodes)
                if np.any(nodes <= 0):
                    continue
                c4 = _compute_c4(nodes)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = nodes.copy()
                    # Refine this point
                    try:
                        res = minimize(
                            _objective_grad,
                            best_roots.copy(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=[(1e-8, None)] * 12,
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                        )
                        if np.all(res.x > 0) and np.isfinite(res.fun):
                            candidate = np.sort(res.x)
                            c42 = _compute_c4(candidate)
                            if c42 < best_c4:
                                best_c4 = c42
                                best_roots = candidate.copy()
                    except Exception:
                        pass
            except Exception:
                continue

    # Final high-precision polish
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
