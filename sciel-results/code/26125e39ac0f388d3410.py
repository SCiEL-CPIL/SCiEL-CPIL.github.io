"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195


def _precompute_I():
    """Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13"""
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n-1] * (2*n - 1) / (2.0 * n)
    return I

_I = _precompute_I()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # I[n] = integral x^{2n} e^{-x^2} dx, precomputed
    I = _I
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    # Build matrix M where M[i,j] = I[i+j], then m_k = p^T M_k p
    # M_k[i,j] = I[k+i+j]
    
    # For efficiency, compute using matrix operations
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    
    m = np.zeros(3)
    for k in range(3):
        s = 0.0
        for i in range(7):
            pi = p[i]
            if pi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += pi * p[j] * I[idx]
        m[k] = s
    
    # Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        s = 0.0
        for i in range(7):
            qi = q[i]
            if qi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += qi * q[j] * I[idx]
        n[k] = 2.0 * np.pi * s
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Precompute S_p[k,i] = sum_j p_j I[k+i+j] and S_q[k,i] = sum_j q_j I[k+i+j]
    # Then dm_k/da_c = 2 * sum_i dp_i[k] * S_p[k,i]
    # And dn_k/da_c = 4*pi * sum_i dq_i[k] * S_q[k,i]
    
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += p[j] * I[idx]
            S_p[k, i] = s
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += q[j] * I[idx]
            S_q[k, i] = s
    
    # dp_k/da_c for each parameter c
    # p0 = a0+3a4+105a8+945a12
    # p1 = -6a4-420a8-7560a12
    # p2 = a4+210a8+13860a12
    # p3 = -28a8-6930a12
    # p4 = a8+1155a12
    # p5 = -66a12
    # p6 = a12
    
    # dp/da0: [1, 0, 0, 0, 0, 0, 0] + 945*da12[0] * [1, -7560/945, 13860/945, -6930/945, 1155/945, -66/945, 1/945]
    # Actually: dp_k/da_c = base_dp_k[c] + (dp_k/da12) * da12[c]
    
    # base dp_k/da_c (without a12 contribution):
    # da0: dp = [1, 0, 0, 0, 0, 0, 0]
    # da4: dp = [3, -6, 1, 0, 0, 0, 0]
    # da8: dp = [105, -420, 210, -28, 1, 0, 0]
    
    # dp_k/da12:
    # [945, -7560, 13860, -6930, 1155, -66, 1]
    
    dp_base = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0]
    ])
    
    dp_a12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dp_base[c] + da12[c] * dp_a12
        
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * np.dot(dp, S_p[k, :])
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 4.0 * np.pi * np.dot(dq, S_q[k, :])
        
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
    t_start = time.time()
    
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
        np.array([-1.3, 0.05, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.4, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        elapsed = time.time() - t_start
        if elapsed > 170:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() - t_start < 180:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if best_params is not None and time.time() - t_start < 188:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
