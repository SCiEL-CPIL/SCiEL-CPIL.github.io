"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Precompute dp/da matrix: dp_k/da_j for k=0..6, j=0..2
# p[0] = 0 (constraint enforced)
# p[1] = -6*a4 - 420*a8 - 7560*a12
# p[2] = a4 + 210*a8 + 13860*a12
# p[3] = -28*a8 - 6930*a12
# p[4] = a8 + 1155*a12
# p[5] = -66*a12
# p[6] = a12
# a12 = -(a0 + 3*a4 + 105*a8)/945
# da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945 = -1/9
_DP_DA = np.zeros((7, 3))
# Column 0: da0
_DP_DA[1, 0] = -7560.0 * (-1.0/945.0)  # 8.0
_DP_DA[2, 0] = 13860.0 * (-1.0/945.0)  # -14.666...
_DP_DA[3, 0] = -6930.0 * (-1.0/945.0)  # 7.333...
_DP_DA[4, 0] = 1155.0 * (-1.0/945.0)   # -1.222...
_DP_DA[5, 0] = -66.0 * (-1.0/945.0)    # 0.0698...
_DP_DA[6, 0] = -1.0/945.0
# Column 1: da4
_DP_DA[1, 1] = -6.0 - 7560.0 * (-3.0/945.0)  # -6 + 24 = 18
_DP_DA[2, 1] = 1.0 + 13860.0 * (-3.0/945.0)  # 1 - 44 = -43
_DP_DA[3, 1] = -6930.0 * (-3.0/945.0)        # 22
_DP_DA[4, 1] = 1155.0 * (-3.0/945.0)         # -3.666...
_DP_DA[5, 1] = -66.0 * (-3.0/945.0)          # 0.2095...
_DP_DA[6, 1] = -3.0/945.0
# Column 2: da8
_DP_DA[1, 2] = -420.0 - 7560.0 * (-105.0/945.0)  # -420 + 840 = 420
_DP_DA[2, 2] = 210.0 + 13860.0 * (-105.0/945.0)  # 210 - 1540 = -1330
_DP_DA[3, 2] = -28.0 - 6930.0 * (-105.0/945.0)   # -28 + 770 = 742
_DP_DA[4, 2] = 1.0 + 1155.0 * (-105.0/945.0)     # 1 - 128.333 = -127.333
_DP_DA[5, 2] = -66.0 * (-105.0/945.0)            # 7.333...
_DP_DA[6, 2] = -105.0/945.0

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.array([
        0.0,  # p[0] = a0 + 3*a4 + 105*a8 + 945*a12 = 0 by construction
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    I = _I
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Precompute S_p[k,i] = sum_j p_j I[k+i+j] and S_q[k,i] = sum_j q_j I[k+i+j]
    # Flatten: S[k*7+i]
    S_p = np.zeros(21)
    S_q = np.zeros(21)
    for i in range(7):
        for k in range(3):
            sp_val = 0.0
            sq_val = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    sp_val += p[j] * I[idx]
                    sq_val += q[j] * I[idx]
            S_p[k*7+i] = sp_val
            S_q[k*7+i] = sq_val
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP_DA[:, c]
        
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                dm[k] += 2.0 * dpi * S_p[k*7+i]
        
        dq = dp * _SIGNS
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                dn[k] += 2.0 * 2.0*np.pi * dqi * S_q[k*7+i]
        
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


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-1.7, 0.06, 0.0005]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.3, 0.01, 0.0001]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 80000, 'xatol': 1e-18, 'fatol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more polish round if time permits
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-20, 'fatol': 1e-22})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
