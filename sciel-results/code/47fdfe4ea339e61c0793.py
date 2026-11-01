"""Analytic uncertainty bound with vectorized gradient + multi-start optimization."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# for n = 0, 1, ..., 13
def _compute_I_table():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        if n == 0:
            I[n] = sp
        else:
            # (2n)! / (4^n n!) * sqrt(pi)
            # = (2n-1)!! / 4^n * sqrt(pi)
            # Compute iteratively: I[n] = I[n-1] * (2n-1) / 4
            I[n] = I[n-1] * (2*n - 1) / 4.0
    return I

_I = _compute_I_table()

# Precompute the coefficient matrix structure
# P(x) = sum_{k=0}^{6} p_k x^{2k}
# p = M @ [a0, a4, a8, a12] where M is 7x4
# a12 = -(a0 + 3*a4 + 105*a8)/945
# So p = M3 @ [a0, a4, a8] + M4 * a12 = (M3 - M4/945 * [1,3,105]) @ [a0,a4,a8]

# M3 columns: coefficients of a0, a4, a8 in p_k (without a12)
# p0: a0*1 + a4*3 + a8*105
# p1: a0*0 + a4*(-6) + a8*(-420)
# p2: a0*0 + a4*1 + a8*210
# p3: a0*0 + a4*0 + a8*(-28)
# p4: a0*0 + a4*0 + a8*1
# p5: a0*0 + a4*0 + a8*0
# p6: a0*0 + a4*0 + a8*0

# M4 column: coefficients of a12 in p_k
# p0: 945, p1: -7560, p2: 13860, p3: -6930, p4: 1155, p5: -66, p6: 1

_M3 = np.array([
    [1.0, 3.0, 105.0],
    [0.0, -6.0, -420.0],
    [0.0, 1.0, 210.0],
    [0.0, 0.0, -28.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

_M4 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# Effective coefficient matrix: p = _Meff @ params where params = [a0, a4, a8]
# a12 = -[1/945, 3/945, 105/945] @ params
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

_Meff = _M3 + np.outer(_M4, _da12)

# For Fourier: q_k = p_k * (-1)^k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Meff_q = _Meff * _signs[:, np.newaxis]

# Precompute I matrix for moment computation
# m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
# This can be written as: m_k = p^T @ A_k @ p where A_k[i,j] = I[k+i+j]
_A_m = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _A_m[k, i, j] = _I[idx]

# For Fourier: n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
_A_n = 2.0 * np.pi * _A_m  # same structure, just scaled

# Gradient of m_k w.r.t. p: 2 * A_k @ p (since m_k = p^T A_k p, d/dp = 2 A_k p for symmetric A_k)
# A_k is symmetric, so this works.


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    params = [a0, a4, a8]
    """
    p = _Meff @ params
    q = _Meff_q @ params
    
    # m_k = p^T @ A_m[k] @ p
    m0 = p @ _A_m[0] @ p
    m2 = p @ _A_m[1] @ p
    m4 = p @ _A_m[2] @ p
    
    n0 = q @ _A_n[0] @ q
    n2 = q @ _A_n[1] @ q
    n4 = q @ _A_n[2] @ q
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient:
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * (A_m[k] @ p) @ _Meff[:, c]
    
    # Precompute A_m[k] @ p and A_n[k] @ q
    Ap0 = _A_m[0] @ p
    Ap1 = _A_m[1] @ p
    Aq0 = _A_n[0] @ q
    Aq1 = _A_n[1] @ q
    
    grad = np.zeros(3)
    for c in range(3):
        # dm0/da_c = 2 * Ap0 @ _Meff[:, c]
        dm0 = 2.0 * np.dot(Ap0, _Meff[:, c])
        dm2 = 2.0 * np.dot(Ap1, _Meff[:, c])
        
        dn0 = 2.0 * np.dot(Aq0, _Meff_q[:, c])
        dn2 = 2.0 * np.dot(Aq1, _Meff_q[:, c])
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.2, 0.05, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
