"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute Gaussian moments I_n = integral x^(2n) e^(-x^2) dx
# I_n = (2n-1)!! / 2^n * sqrt(pi)
# I_0 = sqrt(pi)
# I_n = I_{n-1} * (2n-1)/2
_I_CACHE = None
def _get_I():
    global _I_CACHE
    if _I_CACHE is None:
        _I_CACHE = np.zeros(15) # Need up to index 2*6 + 2 = 14? 
        # p has indices 0..6. i+j max 12. k max 2. idx max 14.
        _I_CACHE[0] = np.sqrt(np.pi)
        for n in range(1, 15):
            _I_CACHE[n] = _I_CACHE[n-1] * (2*n - 1) / 2.0
    return _I_CACHE

# Precompute matrices M_k for m_k = p^T M_k p
# (M_k)_{ij} = I_{k+i+j}
# k=0,1,2
_M_CACHES = None
def _get_M():
    global _M_CACHES
    if _M_CACHES is None:
        I = _get_I()
        _M_CACHES = []
        for k in range(3):
            M = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    M[i, j] = I[k + i + j]
            _M_CACHES.append(M)
    return _M_CACHES

# Precompute matrices N_k for n_k = 2*pi * q^T N_k q
# N_k is same structure as M_k
# Actually n_k = 2*pi * sum q_i q_j I_{k+i+j} = 2*pi * q^T M_k q
# So we can reuse M_k.

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    M = _get_M()
    M0, M1, M2 = M[0], M[1], M[2]
    
    # Moments m_k = p^T M_k p
    m0 = p @ M0 @ p
    m2 = p @ M1 @ p
    m4 = p @ M2 @ p
    
    # Moments n_k = 2*pi * q^T M_k q
    factor = 2.0 * np.pi
    n0 = factor * (q @ M0 @ q)
    n2 = factor * (q @ M1 @ q)
    n4 = factor * (q @ M2 @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # d(m_k)/d(p) = 2 M_k p
    # d(n_k)/d(q) = 2 * factor * M_k q
    # d(vf)/d(p) = (dm2*m0 - m2*dm0) / m0^2
    # d(vF)/d(q) = (dn2*n0 - n2*dn0) / n0^2
    
    # We need gradient w.r.t a0, a4, a8
    # p and q are linear in a0, a4, a8 (and a12 which is linear in them)
    # Let's compute Jacobian of p w.r.t a = [a0, a4, a8]
    # p = J_p @ a + c? No, a12 is dependent.
    # Let's express p directly in terms of a0, a4, a8.
    # p0 = a0 + 3a4 + 105a8 + 945(-(a0+3a4+105a8)/945) = 0. (Wait, p0 is always 0? No, g(0)=0 implies sum c_i H_i(0) = 0.
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945.
    # g(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0.
    # So p0 = g(0) = 0.
    # Let's re-verify p coefficients from H_n expansion.
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # Coeff of x^0: a0 + 3a4 + 105a8 + 945a12 = 0.
    # So p[0] is indeed 0.
    
    # Let's construct J_p (7x3 matrix) such that p = J_p @ a
    J_p = np.zeros((7, 3))
    # p0 = 0 (constraint)
    # p1 (coeff x^2): -6 a4 - 420 a8 - 7560 a12
    # a12 = -1/945 a0 - 3/945 a4 - 105/945 a8
    # p1 = -6 a4 - 420 a8 - 7560 (-1/945 a0 - 3/945 a4 - 105/945 a8)
    # p1 = -6 a4 - 420 a8 + 8 a0 + 24 a4 + 840 a8  (since 7560/945 = 8)
    # p1 = 8 a0 + 18 a4 + 420 a8
    J_p[1, 0] = 8.0
    J_p[1, 1] = 18.0
    J_p[1, 2] = 420.0
    
    # p2 (coeff x^4): a4 + 210 a8 + 13860 a12
    # p2 = a4 + 210 a8 + 13860 (-1/945 a0 - 3/945 a4 - 105/945 a8)
    # 13860/945 = 14.66...? 13860 / 945 = 14.666? 
    # 945 * 10 = 9450. 13860 - 9450 = 4410. 4410 / 945 = 4.66?
    # Let's use fractions or exact division.
    # 13860 / 945 = 132 / 9 = 44 / 3.
    # 13860 * (1/945) = 44/3.
    # 13860 * (3/945) = 44.
    # 13860 * (105/945) = 13860 * (7/63) = 13860/9 = 1540.
    # p2 = a4 + 210 a8 - (44/3) a0 - 44 a4 - 1540 a8
    # p2 = -(44/3) a0 - 43 a4 - 1330 a8
    J_p[2, 0] = -44.0 / 3.0
    J_p[2, 1] = -43.0
    J_p[2, 2] = -1330.0
    
    # p3 (coeff x^6): -28 a8 - 6930 a12
    # 6930 / 945 = 7.333? 6930 / 945 = 198 / 27 = 22 / 3.
    # 6930 * (1/945) = 22/3.
    # 6930 * (3/945) = 22.
    # 6930 * (105/945) = 6930 * (7/63) = 6930/9 = 770.
    # p3 = -28 a8 - (22/3) a0 - 22 a4 - 770 a8
    # p3 = -(22/3) a0 - 22 a4 - 798 a8
    J_p[3, 0] = -22.0 / 3.0
    J_p[3, 1] = -22.0
    J_p[3, 2] = -798.0
    
    # p4 (coeff x^8): a8 + 1155 a12
    # 1155 / 945 = 231 / 189 = 77 / 63 = 11 / 9.
    # 1155 * (1/945) = 11/9.
    # 1155 * (3/945) = 11/3.
    # 1155 * (105/945) = 1155 * (7/63) = 1155/9 = 128.33? 1155/9 = 128.33.
    # 1155 / 9 = 128.333. 1155 = 9 * 128 + 3.
    # 1155 * 105 / 945 = 1155 * 1/9 = 128.333.
    # p4 = a8 - (11/9) a0 - (11/3) a4 - (11/9) a8
    # p4 = -(11/9) a0 - (11/3) a4 - (2/9) a8
    J_p[4, 0] = -11.0 / 9.0
    J_p[4, 1] = -11.0 / 3.0
    J_p[4, 2] = -2.0 / 9.0
    
    # p5 (coeff x^10): -66 a12
    # 66 / 945 = 22 / 315.
    # 66 * 3 / 945 = 22 / 105.
    # 66 * 105 / 945 = 66 / 9 = 22 / 3.
    # p5 = -66 (-1/945 a0 - 3/945 a4 - 105/945 a8)
    # p5 = (66/945) a0 + (198/945) a4 + (6930/945) a8
    # p5 = (22/315) a0 + (22/105) a4 + (22/3) a8
    J_p[5, 0] = 22.0 / 315.0
    J_p[5, 1] = 22.0 / 105.0
    J_p[5, 2] = 22.0 / 3.0
    
    # p6 (coeff x^12): a12
    # p6 = -1/945 a0 - 3/945 a4 - 105/945 a8
    # p6 = -1/945 a0 - 1/315 a4 - 1/9 a8
    J_p[6, 0] = -1.0 / 945.0
    J_p[6, 1] = -1.0 / 315.0
    J_p[6, 2] = -1.0 / 9.0
    
    # q = D @ p where D = diag([1, -1, 1, -1, 1, -1, 1])
    D = np.diag([1, -1, 1, -1, 1, -1, 1])
    J_q = D @ J_p
    
    # Gradient of m_k w.r.t a:
    # dm_k/da = (dp/da)^T (2 M_k p) = J_p^T (2 M_k p)
    # Let G_m_k = 2 M_k p (7x1)
    # dm_k/da (3x1) = J_p^T G_m_k
    
    G_m0 = 2.0 * (M0 @ p)
    G_m1 = 2.0 * (M1 @ p)
    G_m2 = 2.0 * (M2 @ p)
    
    dm0_da = J_p.T @ G_m0
    dm1_da = J_p.T @ G_m1
    
    # Gradient of n_k w.r.t a:
    # dn_k/da = (dq/da)^T (2 factor M_k q) = J_q^T (2 factor M_k q)
    G_n0 = 2.0 * factor * (M0 @ q)
    G_n1 = 2.0 * factor * (M1 @ q)
    
    dn0_da = J_q.T @ G_n0
    dn1_da = J_q.T @ G_n1
    
    # d(vf)/da = (dm1_da * m0 - m2 * dm0_da) / m0^2
    d_vf_da = (dm1_da * m0 - m2 * dm0_da) / (m0 * m0)
    
    # d(vF)/da = (dn1_da * n0 - n2 * dn0_da) / n0^2
    d_vF_da = (dn1_da * n0 - n2 * dn0_da) / (n0 * n0)
    
    # d(bound)/da = (vF * d_vf_da + vf * d_vF_da) / (2 * bound)
    grad = (vF * d_vf_da + vf * d_vF_da) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(50): # More starts now that it's fast
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
    
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
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
# EVOLVE-BLOCK-END
