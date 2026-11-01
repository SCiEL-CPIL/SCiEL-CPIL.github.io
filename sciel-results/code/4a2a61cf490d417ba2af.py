"""Uncertainty inequality: tune Hermite-Gaussian test function to minimize A(f)*A(F)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

def _fact_ratio(n):
    """Compute (2n)! / (4^n * n!) * sqrt(pi) = integral x^{2n} e^{-x^2} dx"""
    if n < 0:
        return 0.0
    if n == 0:
        return math.sqrt(math.pi)
    log_val = 0.0
    for m in range(1, 2*n+1):
        log_val += math.log(m)
    log_val -= n * math.log(4.0)
    for m in range(1, n+1):
        log_val -= math.log(m)
    log_val += 0.5 * math.log(math.pi)
    return math.exp(log_val)

# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..12
_I = np.array([_fact_ratio(n) for n in range(13)], dtype=np.float64)

# Precompute the matrix M[k, i, j] = I[k+i+j] if k+i+j < 13 else 0
# This allows vectorized moment computation
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 13:
                _M[k, i, j] = _I[idx]

# Precompute dp_base and dp_a12 as module-level constants
_dp_base = np.array([
    [1.0, 3.0, 105.0],
    [0.0, -6.0, -420.0],
    [0.0, 1.0, 210.0],
    [0.0, 0.0, -28.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

_dp_a12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _bound_and_grad(params):
    """
    Compute uncertainty bound sqrt(vf * vF) and its analytic gradient.
    Vectorized moment computation.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    # Vectorized: m[k] = sum_{i,j} p[i]*p[j]*M[k,i,j]
    pp = np.outer(p, p)
    m = np.array([np.sum(pp * _M[k]) for k in range(3)])
    
    q = p * _signs
    qq = np.outer(q, q)
    n = np.array([np.sum(qq * _M[k]) for k in range(3)]) * (2.0 * math.pi)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Precompute S_p[k,i] = sum_j p[j] * M[k,i,j] and S_q[k,i] = sum_j q[j] * M[k,i,j]
    S_p = np.array([np.sum(_M[k] * p[None, :], axis=1) for k in range(3)])  # shape (3,7)
    S_q = np.array([np.sum(_M[k] * q[None, :], axis=1) for k in range(3)])  # shape (3,7)
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_base[:, c] + _dp_a12 * _da12[c]
        dm = 2.0 * np.array([np.dot(dp, S_p[k]) for k in range(3)])
        
        dq = dp * _signs
        dn = 4.0 * math.pi * np.array([np.dot(dq, S_q[k]) for k in range(3)])
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    """
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is fixed so that g(0) = 0.
    """
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-17, 'fatol': 1e-19})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish: run L-BFGS-B again from the Nelder-Mead result with very tight tolerances
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params

# EVOLVE-BLOCK-END
