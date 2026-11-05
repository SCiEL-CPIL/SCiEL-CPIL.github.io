"""Direct root optimization for Refined Cohn-Goncalves Laguerre bound.
Uses analytic gradients with respect to roots, mapped through sqrt transform 
to ensure positivity. Multi-start with diverse initializations."""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 80)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _compute_c4_and_grad(roots):
    """
    Computes C4 = I2 / I1^2 and its gradient with respect to roots.
    roots: array of shape (12,)
    Returns: (c4_value, grad_array)
    """
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros_like(roots)

    # Build Q(x) = prod(x - r_i)
    # Using numpy polynomial from roots (ascending order)
    Q = np.poly1d(roots)  # This is descending order, let's stick to ascending for moments
    # np.poly1d is descending. Let's use np.polynomial.polynomial
    Q_asc = np.polynomial.polynomial.polyfromroots(roots)
    
    # P = Q^2
    P = np.convolve(Q_asc, Q_asc)
    # P2 = P^2 = Q^4
    P2 = np.convolve(P, P)
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf, np.zeros_like(roots)
    
    c4 = I2 / (I1 * I1)
    
    # Gradient calculation
    # dI1/dri = -2 * int (Q^2 / (x-ri)) e^-x dx
    # dI2/dri = -4 * int (Q^4 / (x-ri)) e^-x dx
    # Let Si = Q / (x-ri). Then Q^2/(x-ri) = Q * Si.
    # And Q^4/(x-ri) = Q^3 * Si = P * Q * Si.
    
    grad = np.zeros(n)
    
    for i in range(n):
        # Si = Q / (x - ri)
        # Polynomial division: Q_asc / (x - ri)
        # Divisor is [ -ri, 1 ] in ascending order (x - ri = -ri + 1*x)
        divisor = np.array([-roots[i], 1.0])
        Si, _ = np.polynomial.polynomial.polydiv(Q_asc, divisor)
        
        # Integrand for dI1 is Q * Si
        integrand_1 = np.convolve(Q_asc, Si)
        int_1 = _poly_moment(integrand_1)
        
        # Integrand for dI2 is P * Q * Si = P * integrand_1
        # Wait, Q^4/(x-ri) = Q^3 * (Q/(x-ri)) = Q^2 * Q * Si = P * (Q * Si)
        # So integrand_2 = P * integrand_1
        integrand_2 = np.convolve(P, integrand_1)
        int_2 = _poly_moment(integrand_2)
        
        dI1_dr = -2.0 * int_1
        dI2_dr = -4.0 * int_2
        
        # dC4/dri = (dI2 * I1^2 - I2 * 2 * I1 * dI1) / I1^4
        #         = dI2/I1^2 - 2*I2*dI1/I1^3
        grad[i] = dI2_dr / (I1 * I1) - 2.0 * I2 * dI1_dr / (I1 * I1 * I1)
        
    return float(c4), grad

def _objective_transformed(params):
    """
    Objective function for optimization.
    params: array of shape (12,) representing sqrt(roots).
    roots = params^2.
    """
    roots = params ** 2
    
    c4, grad_roots = _compute_c4_and_grad(roots)
    
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(params)
    
    # Chain rule: dC4/dparams = dC4/droots * droots/dparams
    # droots/dparams = 2 * params
    grad_params = grad_roots * 2.0 * params
    
    if not np.all(np.isfinite(grad_params)):
        return 1e10, np.zeros_like(params)
        
    return c4, grad_params

def construct():
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY = 10.0
    
    best_c4 = np.inf
    best_roots = None
    
    def time_ok():
        return (time.time() - start_time) < DEADLINE - SAFETY

    # 1. Initial Guesses
    # Standard Laguerre roots
    x0_lag, _ = roots_laguerre(12)
    x0_lag = np.sort(x0_lag)
    
    # Generalized Laguerre roots for various alpha
    inits = []
    inits.append(x0_lag)
    
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            x0_gen, _ = roots_genlaguerre(12, alpha)
            x0_gen = np.sort(x0_gen)
            if np.all(x0_gen > 0):
                inits.append(x0_gen)
        except Exception:
            pass
            
    # Random perturbations of standard roots
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.standard_normal(12) * 0.1
        x0_pert = x0_lag + noise
        if np.all(x0_pert > 0):
            inits.append(np.sort(x0_pert))

    # 2. Optimization Loop
    for init_roots in inits:
        if not time_ok():
            break
        
        # Transform to parameter space (sqrt)
        init_params = np.sqrt(init_roots)
        
        try:
            res = minimize(
                _objective_transformed,
                init_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res.success or np.isfinite(res.fun):
                cand_params = res.x
                cand_roots = cand_params ** 2
                
                if np.all(cand_roots > 0) and np.isfinite(cand_roots).all():
                    c4_val, _ = _compute_c4_and_grad(cand_roots)
                    if np.isfinite(c4_val) and c4_val < best_c4:
                        best_c4 = c4_val
                        best_roots = cand_roots.copy()
        except Exception:
            pass

    # 3. Polish Phase
    # If we found a good candidate, polish it with higher precision
    if best_roots is not None and time_ok():
        init_params = np.sqrt(best_roots)
        
        # Try a few polish iterations with tighter tolerances
        for polish_iter in range(5):
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_transformed,
                    init_params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                
                cand_params = res.x
                cand_roots = cand_params ** 2
                
                if np.all(cand_roots > 0) and np.isfinite(cand_roots).all():
                    c4_val, _ = _compute_c4_and_grad(cand_roots)
                    if np.isfinite(c4_val) and c4_val < best_c4:
                        best_c4 = c4_val
                        best_roots = cand_roots.copy()
                        init_params = cand_params
                    else:
                        # No improvement, break to save time
                        break
            except Exception:
                break

    # Fallback if optimization failed completely
    if best_roots is None or not np.all(np.isfinite(best_roots)):
        best_roots = x0_lag

    return [float(xi) for xi in best_roots]
