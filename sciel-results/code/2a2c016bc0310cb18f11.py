"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, root
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

# Precompute derivatives of P and P' w.r.t a, b, c (with d dependent)
# d = -(a + 3b + 105c) / 10395
# dP/da = H0 - H12/10395
# dP/db = H4 - 3H12/10395
# dP/dc = H8 - 105H12/10395
_dP_da = _H0 - _H12 / 10395.0
_dP_db = _H4 - 3 * _H12 / 10395.0
_dP_dc = _H8 - 105 * _H12 / 10395.0

# dP'/da = -DH12/10395
# dP'/db = DH4 - 3DH12/10395
# dP'/dc = DH8 - 105DH12/10395
_dPp_da = -_DH12 / 10395.0
_dPp_db = _DH4 - 3 * _DH12 / 10395.0
_dPp_dc = _DH8 - 105 * _DH12 / 10395.0


def _compute_integrals(a, b, c):
    """Compute I0, I2, J0 and their derivatives."""
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    # Derivatives of I0
    dI0_da = 2 * np.dot(_W_I0, P * _dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * _dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * _dP_dc)
    
    # Derivatives of I2
    dI2_da = 2 * np.dot(_W_I2, P * _dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * _dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * _dP_dc)
    
    # Derivatives of J0
    # g_pp = P' - xP
    # dgp_da = dPp_da - x dP_da
    dgp_da = _dPp_da - _X_GQ * _dP_da
    dgp_db = _dPp_db - _X_GQ * _dP_db
    dgp_dc = _dPp_dc - _X_GQ * _dP_dc
    
    dJ0_da = 2 * np.dot(_W_I0, g_prime_poly * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_prime_poly * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_prime_poly * dgp_dc)
    
    return I0, I2, J0, dI0_da, dI0_db, dI0_dc, dI2_da, dI2_db, dI2_dc, dJ0_da, dJ0_db, dJ0_dc


def _objective_and_grad(params):
    """Compute the uncertainty product and its gradient with respect to (a, b, c)."""
    a, b, c = params
    I0, I2, J0, dI0_da, dI0_db, dI0_dc, dI2_da, dI2_db, dI2_dc, dJ0_da, dJ0_db, dJ0_dc = _compute_integrals(a, b, c)
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    product = np.sqrt(I2 * J0) / I0
    
    # Gradient of product
    # d(sqrt(f))/da = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    
    return product, np.array([grad_a, grad_b, grad_c])

def _stationary_eqs_bc(params):
    """
    Solves for stationary points with b=1.
    params = [a, c]
    Returns [dL/da, dL/dc] where L = ln(C)
    """
    a, c = params
    b = 1.0
    I0, I2, J0, dI0_da, dI0_db, dI0_dc, dI2_da, dI2_db, dI2_dc, dJ0_da, dJ0_db, dJ0_dc = _compute_integrals(a, b, c)
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return np.array([1e10, 1e10])
    
    # dL/da = 0.5 * dI2_da/I2 + 0.5 * dJ0_da/J0 - dI0_da/I0
    eq_a = 0.5 * dI2_da / I2 + 0.5 * dJ0_da / J0 - dI0_da / I0
    eq_c = 0.5 * dI2_dc / I2 + 0.5 * dJ0_dc / J0 - dI0_dc / I0
    
    return np.array([eq_a, eq_c])

def construct():
    start_time = time.time()
    deadline = 205.0  # seconds (safety margin before 210s kill)
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    # 1. Find stationary points in the b=1 plane
    # Generate grid of starting points
    start_points = []
    # Around seed (a ~ -1.13, c ~ 0.0003)
    for da in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        for dc in [-0.1, -0.05, 0, 0.05, 0.1]:
            start_points.append([seed_result[0] + da, seed_result[2] + dc])
            
    # Broader grid
    for a in [-5, -3, -2, -1, 0, 1, 2, 3]:
        for c in [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]:
            start_points.append([a, c])
            
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(100):
        start_points.append([rng.randn() * 5, rng.randn() * 0.5])

    solutions = []
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        try:
            res = root(_stationary_eqs_bc, sp, method='hybr', tol=1e-12)
            if res.success:
                a_sol, c_sol = res.x
                # Check if it's a valid point (variances positive)
                I0, I2, J0, *_ = _compute_integrals(a_sol, 1.0, c_sol)
                if I0 > 1e-12 and I2 > 1e-12 and J0 > 1e-12:
                    # Check if this solution is unique (avoid duplicates)
                    is_dup = False
                    for sol in solutions:
                        if np.linalg.norm(np.array([a_sol, c_sol]) - np.array([sol[0], sol[2]])) < 1e-6:
                            is_dup = True
                            break
                    if not is_dup:
                        solutions.append((a_sol, 1.0, c_sol))
        except Exception:
            pass

    # 2. Evaluate all found stationary points
    for sol in solutions:
        score, _ = _objective_and_grad(sol)
        if score < best_score:
            best_score = score
            best_params = np.array(sol)

    # 3. Check b=0 subspace (1D optimization or root finding)
    # If b=0, we can set a=1 (if a!=0) and optimize c, or c=1 and optimize a.
    # Let's just do a local minimization from a few points in b=0 plane
    b0_starts = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [seed_result[0], 0.0, seed_result[2]],
        [-1.0, 0.0, 0.1],
        [0.5, 0.0, -0.1]
    ]
    for sp in b0_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                sp,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # 4. Final polish on best_params
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-14, 'disp': False}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Ensure we don't return worse than seed
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()

    return best_params
# EVOLVE-BLOCK-END
