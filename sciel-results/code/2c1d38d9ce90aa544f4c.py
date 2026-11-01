"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # Safety margin before 210s kill


# Precompute I_n = integral x^{2n} e^{-x^2} dx once
def _precompute_I():
    """Precompute I_n = (2n)! / (4^n n!) * sqrt(pi) for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I_n = (2n-1)/(2n) * I_{n-1} * ... wait, let me just compute directly
            # (2n)! / (4^n n!) = product_{k=1}^{n} (2k-1)/(2k) * (2n)!/(2^n n!) ... 
            # Actually: (2n)! / (4^n n!) = (2n)! / (2^{2n} n!)
            # Let's use: I_n = (2n-1)!! / 2^n * sqrt(pi) / 2^n = (2n-1)!! / 4^n * sqrt(pi)
            # (2n-1)!! = (2n)! / (2^n n!)
            # So I_n = (2n)! / (2^n n!) / 4^n * sqrt(pi) = (2n)! / (2^{3n} n!) * sqrt(pi)
            # Hmm, let me just compute numerically
            log_val = 0.0
            for m in range(1, 2*n+1):
                log_val += np.log(m)
            log_val -= n*np.log(4.0)
            for m in range(1, n+1):
                log_val -= np.log(m)
            log_val += 0.5*np.log(np.pi)
            I[n] = np.exp(log_val)
    return I

_I_CACHE = None

def _get_I():
    global _I_CACHE
    if _I_CACHE is None:
        _I_CACHE = _precompute_I()
    return _I_CACHE

# Precompute coefficient matrices for P(x) in terms of (a0, a4, a8, a12)
# p0 = a0 + 3*a4 + 105*a8 + 945*a12
# p1 = -6*a4 - 420*a8 - 7560*a12
# p2 = a4 + 210*a8 + 13860*a12
# p3 = -28*a8 - 6930*a12
# p4 = a8 + 1155*a12
# p5 = -66*a12
# p6 = a12
#
# With a12 = -(a0 + 3*a4 + 105*a8)/945:
# p0 = a0 + 3*a4 + 105*a8 + 945*(-(a0+3*a4+105*a8)/945) = 0 (as expected, g(0)=0)
# p1 = -6*a4 - 420*a8 - 7560*(-(a0+3*a4+105*a8)/945)
#    = -6*a4 - 420*a8 + (7560/945)*(a0+3*a4+105*a8)
#    = -6*a4 - 420*a8 + 8*(a0+3*a4+105*a8)
#    = 8*a0 + (-6+24)*a4 + (-420+840)*a8
#    = 8*a0 + 18*a4 + 420*a8
# p2 = a4 + 210*a8 + 13860*(-(a0+3*a4+105*a8)/945)
#    = a4 + 210*a8 - (13860/945)*(a0+3*a4+105*a8)
#    = a4 + 210*a8 - 14.666...*(a0+3*a4+105*a8)
#    13860/945 = 14.666... = 44/3
#    = a4 + 210*a8 - (44/3)*a0 - 44*a4 - (44*105/3)*a8
#    = -(44/3)*a0 + (1-44)*a4 + (210 - 1540)*a8
#    = -(44/3)*a0 - 43*a4 - 1330*a8
# p3 = -28*a8 - 6930*(-(a0+3*a4+105*a8)/945)
#    = -28*a8 + (6930/945)*(a0+3*a4+105*a8)
#    6930/945 = 7.333... = 22/3
#    = -28*a8 + (22/3)*a0 + 22*a4 + (22*105/3)*a8
#    = (22/3)*a0 + 22*a4 + (-28 + 770)*a8
#    = (22/3)*a0 + 22*a4 + 742*a8
# p4 = a8 + 1155*(-(a0+3*a4+105*a8)/945)
#    = a8 - (1155/945)*(a0+3*a4+105*a8)
#    1155/945 = 1.2222... = 11/9
#    = a8 - (11/9)*a0 - (11/3)*a4 - (11*105/9)*a8
#    = -(11/9)*a0 - (11/3)*a4 + (1 - 1155/945)*a8
#    = -(11/9)*a0 - (11/3)*a4 + ((945-1155)/945)*a8
#    = -(11/9)*a0 - (11/3)*a4 - (210/945)*a8
#    = -(11/9)*a0 - (11/3)*a4 - (14/63)*a8
#    = -(11/9)*a0 - (11/3)*a4 - (2/9)*a8
# p5 = -66*(-(a0+3*a4+105*a8)/945)
#    = (66/945)*(a0+3*a4+105*a8)
#    66/945 = 22/315
#    = (22/315)*a0 + (66/315)*a4 + (2310/315)*a8
#    = (22/315)*a0 + (22/105)*a4 + (22/3)*a8
# p6 = -(a0+3*a4+105*a8)/945
#    = -(1/945)*a0 - (3/945)*a4 - (105/945)*a8
#    = -(1/945)*a0 - (1/315)*a4 - (1/9)*a8

# Matrix M such that p = M @ [a0, a4, a8]
_M_P = np.array([
    [0.0, 0.0, 0.0],           # p0 = 0
    [8.0, 18.0, 420.0],        # p1
    [-44.0/3.0, -43.0, -1330.0],  # p2
    [22.0/3.0, 22.0, 742.0],   # p3
    [-11.0/9.0, -11.0/3.0, -2.0/9.0],  # p4
    [22.0/315.0, 22.0/105.0, 22.0/3.0],  # p5
    [-1.0/945.0, -1.0/315.0, -1.0/9.0]  # p6
])

# For q_k = p_k * (-1)^k
_q_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_M_Q = _M_P * _q_signs[:, None]

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    I = _get_I()
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute p coefficients
    p = _M_P @ np.array([a0, a4, a8])
    q = _M_Q @ np.array([a0, a4, a8])
    
    # Compute moments m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m0 = 0.0
    m2 = 0.0
    m4 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            m0 += pij * I[i+j]
            m2 += pij * I[i+j+1]
            m4 += pij * I[i+j+2]
    
    # Compute Fourier moments n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n0 = 0.0
    n2 = 0.0
    n4 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            if qij == 0.0:
                continue
            n0 += qij * I[i+j]
            n2 += qij * I[i+j+1]
            n4 += qij * I[i+j+2]
    n0 *= 2.0*np.pi
    n2 *= 2.0*np.pi
    n4 *= 2.0*np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    for c in range(3):
        # dp_k/da_c = _M_P[k, c]
        # dq_k/da_c = _M_Q[k, c]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm0 = 0.0
        dm2 = 0.0
        dm4 = 0.0
        for i in range(7):
            dpi = _M_P[i, c]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            s4 = 0.0
            for j in range(7):
                pj = p[j]
                if pj == 0.0:
                    continue
                s0 += pj * I[i+j]
                s2 += pj * I[i+j+1]
                s4 += pj * I[i+j+2]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
            dm4 += 2.0 * dpi * s4
        
        # dn_k/da_c = 2*2*pi*sum_i dq_i * sum_j q_j I[k+i+j]
        dn0 = 0.0
        dn2 = 0.0
        dn4 = 0.0
        for i in range(7):
            dqi = _M_Q[i, c]
            if dqi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            s4 = 0.0
            for j in range(7):
                qj = q[j]
                if qj == 0.0:
                    continue
                s0 += qj * I[i+j]
                s2 += qj * I[i+j+1]
                s4 += qj * I[i+j+2]
            dn0 += 2.0 * 2.0*np.pi * dqi * s0
            dn2 += 2.0 * 2.0*np.pi * dqi * s2
            dn4 += 2.0 * 2.0*np.pi * dqi * s4
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # More diverse starting points
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
        np.array([-1.7, 0.06, 0.0009]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-4.0, 0.2, 0.003]),
        np.array([0.5, -0.02, 0.0003]),
        np.array([-0.6, 0.025, 0.00035]),
    ]
    
    rng = np.random.default_rng(123)
    # Add random starts with varying scales
    for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        for i in range(5):
            starts.append(rng.normal(0, scale, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 2: Nelder-Mead polish
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more Nelder-Mead if time permits
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
