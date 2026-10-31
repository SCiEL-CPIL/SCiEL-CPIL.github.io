"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# Precompute Gauss-Hermite nodes and weights (15 points is exact for degree <= 29)
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395
# Precompute for derivative: P'(x) terms
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11
# Derivative of H4: 4x^3 - 12x
_DH4 = 4 * _X3 - 12 * _X_GQ
# Derivative of H8: 8x^7 - 168x^5 + 840x^3 - 840x
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
# Derivative of H12: 12x^11 - 840x^9 + 10080x^7 - 75600x^5 + 221760x^3 - 201600x
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals: w_i * 1, w_i * x_i^2
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Precompute derivative-of-P arrays (independent of a,b,c)
_dP_da = _H0 - _H12 / 10395.0
_dP_db = _H4 - 3 * _H12 / 10395.0
_dP_dc = _H8 - 105 * _H12 / 10395.0

# Precompute derivative-of-P_prime arrays
_dPp_da = -_DH12 / 10395.0
_dPp_db = _DH4 - 3 * _DH12 / 10395.0
_dPp_dc = _DH8 - 105 * _DH12 / 10395.0

# Precompute d(g_prime_poly)/d(a,b,c) where g_prime_poly = P_prime - x*P
_dgp_da = _dPp_da - _X_GQ * _dP_da
_dgp_db = _dPp_db - _X_GQ * _dP_db
_dgp_dc = _dPp_dc - _X_GQ * _dP_dc


def _compute_integrals(a, b, c):
    """Compute I0, I2, J0 using precomputed Gauss-Hermite nodes.
    Returns (I0, I2, J0) and also P, P_prime for gradient computation."""
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    P2 = P * P
    I0 = np.dot(_W_I0, P2)
    I2 = np.dot(_W_I2, P2)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    return I0, I2, J0, P, P_prime, d


def _objective_and_grad(params):
    """Compute the uncertainty product and its gradient with respect to (a, b, c)."""
    a, b, c = params
    I0, I2, J0, P, P_prime, d = _compute_integrals(a, b, c)
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    # Precomputed arrays for efficiency
    dP_da = _dP_da
    dP_db = _dP_db
    dP_dc = _dP_dc
    
    # dI0/da = 2 * sum(w * P * dP_da)
    dI0_da = 2.0 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2.0 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2.0 * np.dot(_W_I0, P * dP_dc)
    
    # dI2/da = 2 * sum(w*x^2 * P * dP_da)
    dI2_da = 2.0 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2.0 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2.0 * np.dot(_W_I2, P * dP_dc)
    
    g_pp = P_prime - _X_GQ * P  # current g_prime_poly
    
    dJ0_da = 2.0 * np.dot(_W_I0, g_pp * _dgp_da)
    dJ0_db = 2.0 * np.dot(_W_I0, g_pp * _dgp_db)
    dJ0_dc = 2.0 * np.dot(_W_I0, g_pp * _dgp_dc)
    
    product = np.sqrt(I2 * J0) / I0
    denom = 2.0 * I2 * J0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / denom - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / denom - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / denom - dI0_dc / I0
    
    return product, np.array([grad_a, grad_b, grad_c])


def construct():
    start_time = time.time()
    deadline = 207.0  # seconds (safety margin before 210s kill)
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = [best_params.copy()]
    
    # Perturbed versions of the seed
    for i in range(30):
        perturbation = rng.randn(3) * 0.2
        start_points.append(seed_result + perturbation)
    
    # Random points in reasonable range
    for i in range(30):
        start_points.append(rng.randn(3) * 1.0)
    
    # Structured grid points focused on likely optimum region
    for a_val in np.linspace(-2.0, 0.5, 11):
        for b_val in np.linspace(-0.5, 0.3, 9):
            start_points.append(np.array([a_val, b_val, 0.02]))
    
    # Additional structured points with non-zero c
    for a_val in np.linspace(-1.5, 0.0, 7):
        for b_val in np.linspace(-0.3, 0.3, 7):
            for c_val in np.linspace(-0.05, 0.12, 5):
                start_points.append(np.array([a_val, b_val, c_val]))
    
    # Optimize from each starting point using L-BFGS-B with analytic gradient
    for sp in start_points:
        if time.time() > deadline - 80:
            break
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                sp,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Multi-stage refinement with increasing precision
    # Stage 1: L-BFGS-B with tighter tolerances
    if time.time() < deadline - 60:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Stage 2: BFGS with very tight tolerances
    if time.time() < deadline - 40:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-16, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead local refinement around best point
    if time.time() < deadline - 20:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-17, 'fatol': 1e-19, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Stage 4: Final ultra-precise BFGS polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-18, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_score, _ = _objective_and_grad(best_params)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure we don't return worse than seed
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()
    
    return best_params
# EVOLVE-BLOCK-END
