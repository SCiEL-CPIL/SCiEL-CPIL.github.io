"""Analytic Hermite-Gaussian uncertainty bound with gradient-based optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _moments_and_grads(a0, a4, a8):
    """
    Computes moments and their derivatives with respect to a0, a4, a8.
    a12 is dependent: a12 = -(a0 + 3a4 + 105a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Derivatives of a12
    da12_da0 = -1.0 / 945.0
    da12_da4 = -3.0 / 945.0
    da12_da8 = -105.0 / 945.0

    # Moments of g
    # m0 = sqrt(pi) * (a0^2 + 3a4^2 + 105a8^2 + 945a12^2)
    m0_val = a0*a0 + 3.0*a4*a4 + 105.0*a8*a8 + 945.0*a12*a12
    m0 = np.sqrt(np.pi) * m0_val
    
    dm0_da0 = np.sqrt(np.pi) * (2*a0 + 2*945.0*a12*da12_da0)
    dm0_da4 = np.sqrt(np.pi) * (6*a4 + 2*945.0*a12*da12_da4)
    dm0_da8 = np.sqrt(np.pi) * (210*a8 + 2*945.0*a12*da12_da8)

    # m2 = sqrt(pi) * (3a0^2 + 24a0a4 + 105a4^2 + 1680a4a8 + 2940a8^2 + 11340a8a12)
    # Note: The formula in the original code was:
    # m2 = (a0 * (3.0 * a0 + 12.0 * a4) + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8) + a8 * (840.0 * a4 + 2940.0 * a8) + 11340.0 * a8 * a12) * np.pi**0.5
    # Expanding: 3a0^2 + 12a0a4 + 12a0a4 + 105a4^2 + 840a4a8 + 840a4a8 + 2940a8^2 + 11340a8a12
    # = 3a0^2 + 24a0a4 + 105a4^2 + 1680a4a8 + 2940a8^2 + 11340a8a12
    
    m2_val = 3.0*a0*a0 + 24.0*a0*a4 + 105.0*a4*a4 + 1680.0*a4*a8 + 2940.0*a8*a8 + 11340.0*a8*a12
    m2 = np.sqrt(np.pi) * m2_val
    
    dm2_da0 = np.sqrt(np.pi) * (6.0*a0 + 24.0*a4)
    dm2_da4 = np.sqrt(np.pi) * (24.0*a0 + 210.0*a4 + 1680.0*a8 + 11340.0*a8*da12_da4)
    dm2_da8 = np.sqrt(np.pi) * (1680.0*a4 + 5880.0*a8 + 11340.0*a12 + 11340.0*a8*da12_da8)

    # Moments of G (Fourier transform)
    # g0 = a0 + 3a4 + 105a8 + 945a12
    # g2 = a0 - 3a4 + 105a8 - 945a12
    # g4 = a0 + 3a4 + 105a8 + 945a12 (Same as g0)
    
    g0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    g2 = a0 - 3.0*a4 + 105.0*a8 - 945.0*a12
    g4 = g0
    
    # n0 = 2*pi * (g0^2 + 9g2^2 + 11025g4^2)
    n0_val = g0*g0 + 9.0*g2*g2 + 11025.0*g4*g4
    n0 = 2.0 * np.pi * n0_val
    
    dg0_da0 = 1.0 + 945.0*da12_da0
    dg0_da4 = 3.0 + 945.0*da12_da4
    dg0_da8 = 105.0 + 945.0*da12_da8
    
    dg2_da0 = 1.0 - 945.0*da12_da0
    dg2_da4 = -3.0 - 945.0*da12_da4
    dg2_da8 = 105.0 - 945.0*da12_da8
    
    # dn0/da_i = 2*pi * (2*g0*dg0 + 18*g2*dg2 + 22050*g4*dg4)
    # Note g4=g0, so dg4=dg0
    dn0_da0 = 2.0 * np.pi * (2.0*g0*dg0_da0 + 18.0*g2*dg2_da0 + 22050.0*g0*dg0_da0)
    dn0_da4 = 2.0 * np.pi * (2.0*g0*dg0_da4 + 18.0*g2*dg2_da4 + 22050.0*g0*dg0_da4)
    dn0_da8 = 2.0 * np.pi * (2.0*g0*dg0_da8 + 18.0*g2*dg2_da8 + 22050.0*g0*dg0_da8)

    # n2 = 2*pi * (3g0^2 + 21g2^2 + 2940g4^2)
    n2_val = 3.0*g0*g0 + 21.0*g2*g2 + 2940.0*g4*g4
    n2 = 2.0 * np.pi * n2_val
    
    dn2_da0 = 2.0 * np.pi * (6.0*g0*dg0_da0 + 42.0*g2*dg2_da0 + 5880.0*g0*dg0_da0)
    dn2_da4 = 2.0 * np.pi * (6.0*g0*dg0_da4 + 42.0*g2*dg2_da4 + 5880.0*g0*dg0_da4)
    dn2_da8 = 2.0 * np.pi * (6.0*g0*dg0_da8 + 42.0*g2*dg2_da8 + 5880.0*g0*dg0_da8)

    return m0, m2, n0, n2, (dm0_da0, dm0_da4, dm0_da8), (dm2_da0, dm2_da4, dm2_da8), (dn0_da0, dn0_da4, dn0_da8), (dn2_da0, dn2_da4, dn2_da8)


def _bound_and_grad(params):
    a0, a4, a8 = params
    m0, m2, n0, n2, dm0, dm2, dn0, dn2 = _moments_and_grads(a0, a4, a8)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])
    
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([0.0, 0.0, 0.0])
    
    J = np.sqrt(vf * vF)
    
    # Gradient of J = 0.5 * J * ( (dm2/m2 - dm0/m0) + (dn2/n2 - dn0/n0) )
    # Let A = m2/m0, B = n2/n0. J = sqrt(AB). dJ = 0.5/sqrt(AB) * (B dA + A dB)
    # dA/da_i = (dm2_i/m0 - m2 dm0_i/m0^2) = (1/m0) * (dm2_i - (m2/m0) dm0_i)
    # dB/da_i = (1/n0) * (dn2_i - (n2/n0) dn0_i)
    
    grad = np.zeros(3)
    for i in range(3):
        dA = (dm2[i] * m0 - m2 * dm0[i]) / (m0 * m0)
        dB = (dn2[i] * n0 - n2 * dn0[i]) / (n0 * n0)
        grad[i] = 0.5 * (J / (vf * vF)) * (vF * dA + vf * dB)
        
    return J, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Good starting points
    starts = [
        np.array([-1.13, 0.04, 0.0003]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([1.0, -0.1, 0.0]),
        np.array([-0.5, 0.05, 0.01])
    ]
    
    best_val = 1e18
    best_params = starts[0].copy()
    
    for s in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B', jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    return best_params
# EVOLVE-BLOCK-END
