"""Analytic uncertainty bound with exact gradient + multi-stage L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Precompute index matrices for vectorized moment computation
_NP = 7  # number of polynomial coefficients (degrees 0,2,4,...,12)
_IDX = np.add.outer(np.arange(_NP), np.arange(_NP))  # _IDX[i,j] = i+j
_IDX_K = [k + _IDX for k in range(3)]  # _IDX_K[k][i,j] = k+i+j

# Precompute sign vector for Fourier transform
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute dp/da matrix (derivatives of p w.r.t. a0, a4, a8)
# p[0] = 0 always (by g(0)=0 constraint)
# a12 = -(a0 + 3*a4 + 105*a8)/945
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# dp_k/da_c for each coefficient k and parameter c
# p[0] = a0 + 3a4 + 105a8 + 945*a12 -> always 0, dp0 = [1,3,105] + 945*_DA12 = [0,0,0]
# p[1] = -6a4 - 420a8 - 7560*a12 -> dp1 = [0,-6,-420] - 7560*_DA12
# p[2] = a4 + 210a8 + 13860*a12 -> dp2 = [0,1,210] + 13860*_DA12
# p[3] = -28a8 - 6930*a12 -> dp3 = [0,0,-28] - 6930*_DA12
# p[4] = a8 + 1155*a12 -> dp4 = [0,0,1] + 1155*_DA12
# p[5] = -66*a12 -> dp5 = [0,0,0] - 66*_DA12
# p[6] = a12 -> dp6 = [0,0,0] + 1*_DA12

_DPDA = np.zeros((3, _NP))
_DPDA[0, 0] = 1.0 + 945.0*_DA12[0]  # = 0
_DPDA[1, 0] = 3.0 + 945.0*_DA12[1]  # = 0
_DPDA[2, 0] = 105.0 + 945.0*_DA12[2]  # = 0
_DPDA[0, 1] = 0.0 - 7560.0*_DA12[0]
_DPDA[1, 1] = -6.0 - 7560.0*_DA12[1]
_DPDA[2, 1] = -420.0 - 7560.0*_DA12[2]
_DPDA[0, 2] = 0.0 + 13860.0*_DA12[0]
_DPDA[1, 2] = 1.0 + 13860.0*_DA12[1]
_DPDA[2, 2] = 210.0 + 13860.0*_DA12[2]
_DPDA[0, 3] = 0.0 - 6930.0*_DA12[0]
_DPDA[1, 3] = 0.0 - 6930.0*_DA12[1]
_DPDA[2, 3] = -28.0 - 6930.0*_DA12[2]
_DPDA[0, 4] = 0.0 + 1155.0*_DA12[0]
_DPDA[1, 4] = 0.0 + 1155.0*_DA12[1]
_DPDA[2, 4] = 1.0 + 1155.0*_DA12[2]
_DPDA[0, 5] = 0.0 - 66.0*_DA12[0]
_DPDA[1, 5] = 0.0 - 66.0*_DA12[1]
_DPDA[2, 5] = 0.0 - 66.0*_DA12[2]
_DPDA[0, 6] = 0.0 + 1.0*_DA12[0]
_DPDA[1, 6] = 0.0 + 1.0*_DA12[1]
_DPDA[2, 6] = 0.0 + 1.0*_DA12[2]


def _bound_and_grad(params):
    """
    Compute uncertainty bound and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.empty(_NP)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    
    # Vectorized moment computation
    m = np.empty(3)
    for k in range(3):
        m[k] = np.sum(pp * I[_IDX_K[k]])
    
    # Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    qq = np.outer(q, q)
    n = np.empty(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[_IDX_K[k]])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Precompute S_p[k,i] = sum_j p_j I[k+i+j] and S_q[k,i] = sum_j q_j I[k+i+j]
    S_p = np.empty((3, _NP))
    S_q = np.empty((3, _NP))
    for k in range(3):
        # S_p[k,i] = sum_j p[j] * I[k+i+j] = p @ I[_IDX_K[k][:, i]]
        # Vectorized: for each i, sum over j of p[j] * I[k+i+j]
        for i in range(_NP):
            S_p[k, i] = np.dot(p, I[_IDX_K[k][i, :]])
            S_q[k, i] = np.dot(q, I[_IDX_K[k][i, :]])
    
    # Compute gradient
    grad = np.empty(3)
    for c in range(3):
        dp = _DPDA[c]
        
        # dm[k] = 2 * sum_i dp[i] * S_p[k,i]
        dm = 2.0 * (dp @ S_p)
        
        # dq = dp * signs
        dq = dp * _SIGNS
        # dn[k] = 4*pi * sum_i dq[i] * S_q[k,i]
        dn = 4.0 * np.pi * (dq @ S_q)
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _bound_only(params):
    """Compute just the bound value (no gradient) for fast evaluation."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(_NP)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    
    m0 = 0.0
    m2 = 0.0
    for k in range(2):
        m0 += np.sum(pp * I[_IDX_K[0]])
        m2 += np.sum(pp * I[_IDX_K[1]])
    
    q = p * _SIGNS
    qq = np.outer(q, q)
    
    n0 = 0.0
    n2 = 0.0
    for k in range(2):
        n0 += 2.0 * np.pi * np.sum(qq * I[_IDX_K[0]])
        n2 += 2.0 * np.pi * np.sum(qq * I[_IDX_K[1]])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def construct():
    deadline = time.time() + 200
    
    # Exploit homogeneity: bound depends only on direction of (a0, a4, a8)
    # Fix a0 = 1 and search over (a4, a8)
    # Use a dense grid + Brent refinement in 2D
    
    best_val = np.inf
    best_params = None
    
    # Stage 1: Dense grid search over (a4, a8) with a0 = 1
    # Based on seed: a0=-1.13, a4=0.04, a8=0.000308642
    # Ratios: a4/a0 = -0.0354, a8/a0 = -0.000273
    # Search around these ratios
    
    a4_range = np.linspace(-0.2, 0.2, 200)
    a8_range = np.linspace(-0.005, 0.005, 200)
    
    grid_best = np.inf
    grid_best_a4 = 0.0
    grid_best_a8 = 0.0
    
    for a4 in a4_range:
        if time.time() > deadline - 120:
            break
        for a8 in a8_range:
            params = np.array([1.0, a4, a8])
            val = _bound_only(params)
            if val < grid_best:
                grid_best = val
                grid_best_a4 = a4
                grid_best_a8 = a8
    
    if grid_best < best_val:
        best_val = grid_best
        best_params = np.array([1.0, grid_best_a4, grid_best_a8])
    
    # Stage 2: L-BFGS-B from the grid best point
    if best_params is not None and time.time() < deadline - 60:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: Additional L-BFGS-B from seed and other good starts
    extra_starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([1.0, -0.035, -0.0003]),
        np.array([1.0, -0.04, -0.00035]),
        np.array([1.0, -0.03, -0.00025]),
    ]
    
    for start in extra_starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Stage 4: High-precision polish
    if time.time() < deadline - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 5: Nelder-Mead refinement
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
