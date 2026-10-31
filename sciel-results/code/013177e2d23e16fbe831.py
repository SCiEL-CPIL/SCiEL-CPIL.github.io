"""Optimized program: analytic-gradient L-BFGS-B with multi-start and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = 205.0  # safety margin before 210s kill

# Precompute constants
_DEGREES = np.array([0, 4, 8, 12])
_N = 4

# Precompute 2^k * k! for each degree
_NORM_FACTORS = np.array([float(2**k * factorial(k)) for k in _DEGREES]) * sqrt(pi)

# Precompute the M_x matrix: M_x[i,j] = ∫ H_i(x) x^2 H_j(x) exp(-x^2) dx
# Using: x^2 H_j = (1/2) H_{j+2} + (2j+1) H_j + (j(j-1)/2) H_{j-2}
# M_x[i,j] = (1/2)*δ_{i,j+2}*2^j*j!*√π + (2j+1)*δ_{i,j}*2^j*j!*√π + (j(j-1)/2)*δ_{i,j-2}*2^j*j!*√π
_MX = np.zeros((_N, _N))
for ii in range(_N):
    for jj in range(_N):
        i_deg = _DEGREES[ii]
        j_deg = _DEGREES[jj]
        nf = float(2**j_deg * factorial(j_deg)) * sqrt(pi)
        term = 0.0
        if i_deg == j_deg + 2:
            term += 0.5 * nf
        if i_deg == j_deg:
            term += (2 * j_deg + 1) * nf
        if i_deg == j_deg - 2 and j_deg >= 2:
            term += (j_deg * (j_deg - 1) / 2.0) * nf
        _MX[ii, jj] = term

# Precompute the M_p matrix for momentum space
# F(g)(p) = sum c_k * i^k * H_k(p) exp(-p^2/2)
# <p^2> = (1/||g||^2) sum_{i,j} c_i * c_j * i^(i+j) * M_x[i,j]
# For degrees {0,4,8,12}: i^k = 1 for k=0,4,8,12 (since i^4=1)
# So i^(i+j) = 1 for all pairs in our set
# Therefore M_p = M_x
_MP = _MX.copy()

# Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 544320*c12 = 0
# => c12 = -(c0 + 12*c4 + 1680*c8) / 544320
_CONST = np.array([1.0, 12.0, 1680.0, 544320.0])
_C12_DENOM = 544320.0

def _get_full_coeffs(params):
    """Get all 4 coefficients from the 3 free parameters."""
    c0, c4, c8 = params
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / _C12_DENOM
    return np.array([c0, c4, c8, c12])

def _compute_bound(params):
    """Compute the upper bound A(f)*A(F(f)) for given 3 free coefficients."""
    coeffs = _get_full_coeffs(params)
    
    # ||g||^2 = sum c_k^2 * norm_factor_k
    norm_sq = np.sum(coeffs**2 * _NORM_FACTORS)
    if norm_sq <= 1e-30:
        return np.inf
    
    # <x^2> = coeffs^T @ M_x @ coeffs / norm_sq
    x2_moment = coeffs @ _MX @ coeffs
    x2_var = x2_moment / norm_sq
    
    # <p^2> = coeffs^T @ M_p @ coeffs / norm_sq
    p2_moment = coeffs @ _MP @ coeffs
    p2_var = p2_moment / norm_sq
    
    return x2_var * p2_var

def _compute_bound_and_grad(params):
    """Compute bound and gradient analytically."""
    coeffs = _get_full_coeffs(params)
    
    norm_sq = np.sum(coeffs**2 * _NORM_FACTORS)
    if norm_sq <= 1e-30:
        return np.inf, np.zeros(3)
    
    x2_moment = coeffs @ _MX @ coeffs
    p2_moment = coeffs @ _MP @ coeffs
    x2_var = x2_moment / norm_sq
    p2_var = p2_moment / norm_sq
    
    bound = x2_var * p2_var
    
    # Gradient with respect to the 3 free parameters
    # coeffs = [c0, c4, c8, c12] where c12 = -(c0 + 12*c4 + 1680*c8)/544320
    # d(coeffs)/d(params) is a 4x3 matrix
    J = np.zeros((_N, 3))
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0
    J[3, 0] = -1.0 / _C12_DENOM
    J[3, 1] = -12.0 / _C12_DENOM
    J[3, 2] = -1680.0 / _C12_DENOM
    
    # d(bound)/d(coeffs) where bound = (c^T Mx c / N) * (c^T Mp c / N)
    # = (1/N^2) * (c^T Mx c) * (c^T Mp c)
    # d/dc = (1/N^2) * [ (c^T Mp c) * 2*Mx*c + (c^T Mx c) * 2*Mp*c ]
    #        - (2/N^3) * N' * (c^T Mx c) * (c^T Mp c)  where N' = 2*c^T*diag(NF)*c
    # Actually let's do it more carefully.
    # Let f = c^T Mx c, g = c^T Mp c, N = c^T D c where D = diag(NF)
    # bound = f*g / N^2
    # d(bound)/dc = (g * df/dc * N - f * dg/dc * N - f*g * 2*N * dN/dc) / N^3
    #             = g * df/dc / N^2 - f * dg/dc / N^2 - 2*f*g*dN/dc / N^3
    
    df_dc = 2.0 * _MX @ coeffs
    dg_dc = 2.0 * _MP @ coeffs
    dN_dc = 2.0 * coeffs * _NORM_FACTORS
    
    d_bound_dc = (g * df_dc - f * dg_dc) / (norm_sq**2) - 2.0 * f * g * dN_dc / (norm_sq**3)
    
    # Chain rule: d(bound)/d(params) = J^T @ d_bound_dc
    grad = J.T @ d_bound_dc
    
    return bound, grad

def _objective(params):
    return _compute_bound(params)

def _objective_grad(params):
    bound, grad = _compute_bound_and_grad(params)
    return bound, grad

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate seed
    seed_score = _compute_bound(best_coeffs)
    best_score = seed_score
    
    # Multi-start L-BFGS-B with analytic gradient
    starts = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -0.1, 0.001]),
        np.array([-1.0, 0.1, -0.001]),
        np.array([0.5, -0.05, 0.0005]),
        np.array([-0.5, 0.05, -0.0005]),
        np.array([2.0, -0.2, 0.002]),
        np.array([-2.0, 0.2, -0.002]),
    ]
    
    for x0 in starts:
        if time.time() > _DEADLINE - 5.0:
            break
        try:
            result = minimize(_objective_grad, x0, method='L-BFGS-B',
                             jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if time.time() < _DEADLINE - 5.0:
        try:
            result = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                             jac=True, options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
