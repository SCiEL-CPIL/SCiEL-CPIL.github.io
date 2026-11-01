"""Analytic uncertainty bound with precomputed structure + analytic Hessian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 190  # safety margin before 210s kill

def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
    H_0 = 1
    H_4 = x^4 - 6x^2 + 3
    H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    We represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    H_0: p = [1, 0, 0, 0, 0, 0, 0]
    H_4: p = [3, -6, 1, 0, 0, 0, 0]
    H_8: p = [105, -420, 210, -28, 1, 0, 0]
    H_12: p = [945, -7560, 13860, -6930, 1155, -66, 1]
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H

def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12 (need up to x^{24} since max degree is 12, |P|^2 has degree 24)
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I[n] = I[n-1] * (2n-1) / (4*n)
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I

_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute matrices for moments
# m0 = sum pi pj I[i+j]
# m2 = sum pi pj I[1+i+j]
# n0 = 2*pi sum qi qj I[i+j]
# n2 = 2*pi sum qi qj I[1+i+j]

# M0_p[i,j] = I[i+j]
# M2_p[i,j] = I[1+i+j]
# M0_q[i,j] = 2*pi * I[i+j]
# M2_q[i,j] = 2*pi * I[1+i+j]

_idx = np.arange(7)
_M0_p = _I[_idx[:, None] + _idx[None, :]]
_M2_p = _I[1 + _idx[:, None] + _idx[None, :]]
_M0_q = 2.0 * np.pi * _M0_p
_M2_q = 2.0 * np.pi * _M2_p

# The constraint P(0)=0 implies a0 + 3a4 + 105a8 + 945a12 = 0 => a12 = -(a0 + 3a4 + 105a8)/945
# p = a0*H0 + a4*H4 + a8*H8 + a12*H12
# p = a0*(H0 - H12/945) + a4*(H4 - 3*H12/945) + a8*(H8 - 105*H12/945)
# Let's define the effective basis vectors for a0, a4, a8
# v0 = H0 - H12/945
# v4 = H4 - 3*H12/945
# v8 = H8 - 105*H12/945
_v0 = _H[0] - _H[3] / 945.0
_v4 = _H[1] - 3.0 * _H[3] / 945.0
_v8 = _H[2] - 105.0 * _H[3] / 945.0

# Matrix V where columns are v0, v4, v8. Shape (7, 3)
_V = np.column_stack([_v0, _v4, _v8])

# Hessian of moments w.r.t a:
# m0(a) = (V a)^T M0_p (V a) = a^T (V^T M0_p V) a
# H_m0 = 2 * V^T M0_p V
# H_m2 = 2 * V^T M2_p V
# H_n0 = 2 * V^T M0_q V
# H_n2 = 2 * V^T M2_q V
# Note: The factor 2 comes from the derivative of quadratic form x^T A x which is 2Ax (grad) and 2A (hess).
# Wait, m = p^T M p. dp/da = V. dm/da = 2 V^T M p = 2 V^T M V a. d2m/da2 = 2 V^T M V.

_H_m0 = 2.0 * (_V.T @ _M0_p @ _V)
_H_m2 = 2.0 * (_V.T @ _M2_p @ _V)
_H_n0 = 2.0 * (_V.T @ _M0_q @ _V)
_H_n2 = 2.0 * (_V.T @ _M2_q @ _V)

def _eval_moments(a):
    """Evaluate m0, m2, n0, n2 and their gradients and Hessians."""
    # p = V a
    p = _V @ a
    
    # Values
    m0 = p @ _M0_p @ p
    m2 = p @ _M2_p @ p
    
    q = p * _SIGN
    n0 = q @ _M0_q @ q # Note: _M0_q already includes 2*pi
    n2 = q @ _M2_q @ q
    
    # Gradients: 2 * H_matrix @ a
    gm0 = _H_m0 @ a
    gm2 = _H_m2 @ a
    gn0 = _H_n0 @ a
    gn2 = _H_n2 @ a
    
    return m0, m2, n0, n2, gm0, gm2, gn0, gn2

def _objective_grad_hess(a):
    """
    Compute objective, gradient, and Hessian.
    J = sqrt(vf * vF) where vf = m2/m0, vF = n2/n0.
    """
    m0, m2, n0, n2, gm0, gm2, gn0, gn2 = _eval_moments(a)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18]), np.eye(3) * 1e18
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18]), np.eye(3) * 1e18
        
    J = np.sqrt(vf * vF)
    
    # Log derivatives
    # d ln J = 0.5 * (d ln m2 - d ln m0 + d ln n2 - d ln n0)
    # d ln m2 = gm2 / m2
    # d ln m0 = gm0 / m0
    # etc.
    
    dln_m2 = gm2 / m2
    dln_m0 = gm0 / m0
    dln_n2 = gn2 / n2
    dln_n0 = gn0 / n0
    
    dln_J = 0.5 * (dln_m2 - dln_m0 + dln_n2 - dln_n0)
    
    grad_J = J * dln_J
    
    # Hessian of J
    # H_J = J * (H_dln_J + outer(dln_J, dln_J))
    
    # H_dln_m2 = H_m2 / m2 - outer(gm2, gm2) / m2^2
    # H_dln_m0 = H_m0 / m0 - outer(gm0, gm0) / m0^2
    # H_dln_n2 = H_n2 / n2 - outer(gn2, gn2) / n2^2
    # H_dln_n0 = H_n0 / n0 - outer(gn0, gn0) / n0^2
    
    H_dln_m2 = _H_m2 / m2 - np.outer(gm2, gm2) / (m2 * m2)
    H_dln_m0 = _H_m0 / m0 - np.outer(gm0, gm0) / (m0 * m0)
    H_dln_n2 = _H_n2 / n2 - np.outer(gn2, gn2) / (n2 * n2)
    H_dln_n0 = _H_n0 / n0 - np.outer(gn0, gn0) / (n0 * n0)
    
    H_dln_J = 0.5 * (H_dln_m2 - H_dln_m0 + H_dln_n2 - H_dln_n0)
    
    H_J = J * (H_dln_J + np.outer(dln_J, dln_J))
    
    # Symmetrize to remove numerical noise
    H_J = (H_J + H_J.T) / 2.0
    
    return J, grad_J, H_J

def _objective(a):
    val, _, _ = _objective_grad_hess(a)
    return val

def _objective_grad(a):
    _, grad, _ = _objective_grad_hess(a)
    return grad

def _objective_hess(a):
    _, _, hess = _objective_grad_hess(a)
    return hess

def _objective_grad_hess_wrapper(a):
    return _objective_grad_hess(a)

def construct():
    """
    Optimizes the coefficients using Newton-CG with analytic Hessian.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Initial best from seed
    try:
        v, g, h = _objective_grad_hess(best_params)
        if np.isfinite(v) and v < best_val:
            best_val = v
    except Exception:
        pass

    # Starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.3, 0.03, 0.00025]),
        np.array([-0.9, 0.07, 0.0009]),
        np.array([-1.7, 0.02, 0.00015]),
    ]
    
    rng = np.random.default_rng(42)
    # Add random starts in a reasonable box
    for _ in range(50):
        starts.append(rng.normal([0, 0.04, 0.0005], [1.0, 0.1, 0.002]))

    # Sort starts by initial objective value to prioritize good ones
    starts_with_vals = []
    for s in starts:
        try:
            v = _objective(s)
            if np.isfinite(v):
                starts_with_vals.append((v, s))
        except Exception:
            pass
    
    starts_with_vals.sort(key=lambda x: x[0])
    
    # Optimize from top starts
    for i, (v0, s) in enumerate(starts_with_vals):
        if time.time() > _DEADLINE - 20:
            break
        
        # Use Newton-CG for fast convergence
        try:
            res = minimize(_objective, s, jac=_objective_grad, hess=_objective_hess,
                          method='Newton-CG',
                          options={'maxiter': 1000, 'disp': False, 'maxls': 100, 'gtol': 1e-16, 'xtol': 1e-16})
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
                
                # Polish with L-BFGS-B if time permits, just in case Newton-CG overshoots or has issues
                if time.time() < _DEADLINE - 10 and i < 10: # Only polish top few
                    res2 = minimize(_objective, best_params, jac=_objective_grad,
                                  method='L-BFGS-B',
                                  options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16})
                    if res2.fun < best_val and np.isfinite(res2.fun):
                        best_val = res2.fun
                        best_params = res2.x.copy()
                        
        except Exception:
            pass

    # Final polish with high precision
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, jac=_objective_grad, hess=_objective_hess,
                          method='Newton-CG',
                          options={'maxiter': 2000, 'disp': False, 'maxls': 200, 'gtol': 1e-18, 'xtol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
