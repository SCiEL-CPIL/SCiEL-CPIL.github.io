"""
Optimized program for the uncertainty inequality task.
Uses scale invariance to reduce the 3D search to a 2D search, 
which is more robust and allows for a finer global search.
"""

import numpy as np
from scipy.optimize import minimize
import time

# Precompute Gauss-Hermite nodes and weights
# 30 points is exact for polynomials of degree up to 59, which is plenty for H12^2 (degree 24)
_N_GQ = 30
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ

# Precompute Hermite polynomials H0, H4, H8, H12 at the nodes
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Precompute derivatives for gradient computation if needed, 
# but for the 2D reduction we will use numerical gradients or 
# rely on the smoothness for Nelder-Mead/Powell.
# Actually, analytic gradients are faster. Let's compute them for the 3D case 
# and use them in the 2D wrapper.

# Derivatives of H4, H8, H12
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2


def _compute_bound_and_grad_3d(a, b, c):
    """
    Compute the uncertainty bound and its gradient for (a, b, c).
    Returns (score, grad).
    """
    # d is determined by P(0)=0 => a + 3b + 105c + 10395d = 0
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    score = np.sqrt(I2 * J0) / I0
    
    # Gradient computation
    dP_da = _H0 - _H12 / 10395.0
    dP_db = _H4 - 3 * _H12 / 10395.0
    dP_dc = _H8 - 105 * _H12 / 10395.0
    
    dI0_da = 2 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * dP_dc)
    
    dI2_da = 2 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * dP_dc)
    
    dPp_da = -_DH12 / 10395.0
    dPp_db = _DH4 - 3 * _DH12 / 10395.0
    dPp_dc = _DH8 - 105 * _DH12 / 10395.0
    
    g_pp = P_prime - _X_GQ * P
    
    dgp_da = dPp_da - _X_GQ * dP_da
    dgp_db = dPp_db - _X_GQ * dP_db
    dgp_dc = dPp_dc - _X_GQ * dP_dc
    
    dJ0_da = 2 * np.dot(_W_I0, g_pp * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * dgp_dc)
    
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    
    return score, np.array([grad_a, grad_b, grad_c])


def _bound_2d(params):
    """
    Objective function for the 2D reduction where a=1.
    params = [u, v] corresponding to b=u, c=v.
    """
    u, v = params
    score, _ = _compute_bound_and_grad_3d(1.0, u, v)
    return score


def _grad_2d(params):
    """
    Gradient for the 2D reduction.
    """
    u, v = params
    _, grad = _compute_bound_and_grad_3d(1.0, u, v)
    # grad is [da, db, dc]. We need [du, dv] which are [db, dc]
    return np.array([grad[1], grad[2]])


def _bound_a0(params):
    """
    Objective function for the a=0 case, where b=1.
    params = [v] corresponding to c=v.
    """
    v = params[0]
    score, _ = _compute_bound_and_grad_3d(0.0, 1.0, v)
    return score


def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin
    
    best_score = 1e10
    best_params_3d = np.array([-113 / 100, 1 / 25, 1 / 3240])  # Seed
    
    # 1. Optimize the a=1 sector (2D problem)
    # Grid search
    u_range = np.linspace(-2.0, 2.0, 41)
    v_range = np.linspace(-2.0, 2.0, 41)
    
    candidates_2d = []
    
    for u in u_range:
        for v in v_range:
            if time.time() > deadline - 60:
                break
            try:
                score = _bound_2d([u, v])
                if score < best_score:
                    best_score = score
                    best_params_3d = np.array([1.0, u, v])
                candidates_2d.append((score, u, v))
            except Exception:
                pass
        if time.time() > deadline - 60:
            break
    
    # Refine top candidates
    if time.time() < deadline - 30:
        candidates_2d.sort(key=lambda x: x[0])
        for _, u, v in candidates_2d[:20]:
            try:
                result = minimize(
                    _bound_2d,
                    np.array([u, v]),
                    jac=_grad_2d,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params_3d = np.array([1.0, result.x[0], result.x[1]])
            except Exception:
                pass
    
    # 2. Optimize the a=0 sector (1D problem)
    # Grid search
    v_range_a0 = np.linspace(-2.0, 2.0, 41)
    candidates_a0 = []
    
    for v in v_range_a0:
        if time.time() > deadline - 30:
            break
        try:
            score = _bound_a0([v])
            if score < best_score:
                best_score = score
                best_params_3d = np.array([0.0, 1.0, v])
            candidates_a0.append((score, v))
        except Exception:
            pass
    
    if time.time() < deadline - 10:
        candidates_a0.sort(key=lambda x: x[0])
        for _, v in candidates_a0[:10]:
            try:
                result = minimize(
                    _bound_a0,
                    np.array([v]),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params_3d = np.array([0.0, 1.0, result.x[0]])
            except Exception:
                pass
    
    # 3. Final polish on the best 3D point found
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _compute_bound_and_grad_3d(x[0], x[1], x[2])[0],
                best_params_3d,
                jac=lambda x: _compute_bound_and_grad_3d(x[0], x[1], x[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params_3d = result.x
        except Exception:
            pass
    
    # Ensure we don't return worse than seed
    seed_score, _ = _compute_bound_and_grad_3d(-113 / 100, 1 / 25, 1 / 3240)
    if best_score > seed_score:
        best_params_3d = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return best_params_3d
