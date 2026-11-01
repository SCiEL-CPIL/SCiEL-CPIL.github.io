"""Optimized uncertainty bound with multi-start L-BFGS-B and high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # Safety margin before 210s kill


def _precompute_I():
    """Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! / 4^n * sqrt(pi)"""
    sp = np.sqrt(np.pi)
    I = np.zeros(15)  # Need up to I[12] for p_i p_j with i,j up to 6
    I[0] = sp
    for n in range(1, 15):
        I[n] = I[n-1] * (2*n - 1) / (4*n)
    return I


_I_TABLE = _precompute_I()

# Precompute derivative of p with respect to each parameter
# p_k = sum of contributions from a0, a4, a8, a12
# a12 = -(a0 + 3*a4 + 105*a8)/945
# dp_k/da_c for c in {0,1,2} corresponding to a0, a4, a8

# Coefficients for each Hermite polynomial in power form:
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 1680  (wait, let me use standard)
# Actually, physicists' Hermite: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H0 = 1
# H1 = 2x
# H2 = 4x^2 - 2
# H3 = 8x^3 - 12x
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 4608x^6 + 20160x^4 - 17280x^2 + 1680
# H12 = 4096x^12 - 147456x^10 + 1290240x^8 - 3674880x^6 + 2634240x^4 - 3342720x^2 + 1032192

# But we only need even powers since g is even. Let me redefine:
# g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
# For even functions, we can write P(x) = sum_{k=0}^{6} p_k * x^{2k}

# H0 = 1 = 1*x^0
# H4 = 16x^4 - 48x^2 + 12 = 12*x^0 - 48*x^2 + 16*x^4
# H8 = 256x^8 - 4608x^6 + 20160x^4 - 17280x^2 + 1680 = 1680*x^0 - 17280*x^2 + 20160*x^4 - 4608*x^6 + 256*x^8
# H12 = 4096x^12 - 147456x^10 + 1290240x^8 - 3674880x^6 + 2634240x^4 - 3342720x^2 + 1032192

# So:
# p0 = a0 + 12*a4 + 1680*a8 + 1032192*a12
# p1 = -48*a4 - 17280*a8 - 3342720*a12
# p2 = 16*a4 + 20160*a8 + 2634240*a12
# p3 = -4608*a8 - 3674880*a12
# p4 = 256*a8 + 1290240*a12
# p5 = -147456*a12
# p6 = 4096*a12

# a12 = -(a0 + 3*a4 + 105*a8)/945

# Let me verify H4(0): H4(0) = 12. And from the constraint g(0) = 0:
# a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# a0*1 + a4*12 + a8*1680 + a12*1032192 = 0
# a12 = -(a0 + 12*a4 + 1680*a8)/1032192

# But the problem states a12 = -(a0+3a4+105a8)/945. Let me check: 1032192/945 = 1092.27... that's not integer.
# 1032192/945 = 1092.27... Hmm. Let me recompute.

# Actually, I think the Hermite polynomials might be defined differently. Let me use the standard probabilists' or check the ratio.
# 1032192 / 945 = 1092.27... not clean.
# 1680 / 105 = 16
# 12 / 3 = 4
# So the ratios are 1:4:16 for H0:H4:H8 at x=0, and H12(0) should be such that 1032192/945 is clean.
# 945 * 1092 = 1031940, not 1032192.
# Let me just use the given constraint: a12 = -(a0 + 3*a4 + 105*a8)/945

# This means the Hermite polynomials are scaled such that:
# H0(0) = 1, H4(0) = 3, H8(0) = 105, H12(0) = 945
# So these are likely normalized differently. Let me just use the given relation.

# So p_k in terms of a0, a4, a8, a12 where a12 = -(a0+3a4+105a8)/945:
# The exact form of P(x) in powers of x^2 needs to be determined from the Hermite polynomials with these values at 0.

# Given the constraint, let me just define the mapping from (a0,a4,a8) to p vector directly.

# From the original code, the mapping is:
# p[0] = a0 + 3*a4 + 105*a8 + 945*a12
# p[1] = -6*a4 - 420*a8 - 7560*a12
# p[2] = a4 + 210*a8 + 13860*a12
# p[3] = -28*a8 - 6930*a12
# p[4] = a8 + 1155*a12
# p[5] = -66*a12
# p[6] = a12

# With a12 = -(a0+3a4+105a8)/945, we can substitute:
# p[0] = a0 + 3a4 + 105a8 + 945*(-(a0+3a4+105a8)/945) = a0+3a4+105a8 - (a0+3a4+105a8) = 0
# So p[0] = 0 always! That makes sense since g(0) = 0.

# p[1] = -6a4 - 420a8 - 7560*(-(a0+3a4+105a8)/945) = -6a4 - 420a8 + (7560/945)*(a0+3a4+105a8)
# 7560/945 = 8
# p[1] = -6a4 - 420a8 + 8*(a0+3a4+105a8) = -6a4 - 420a8 + 8a0 + 24a4 + 840a8 = 8a0 + 18a4 + 420a8

# p[2] = a4 + 210a8 + 13860*(-(a0+3a4+105a8)/945) = a4 + 210a8 - (13860/945)*(a0+3a4+105a8)
# 13860/945 = 14.666... = 44/3
# p[2] = a4 + 210a8 - (44/3)*(a0+3a4+105a8) = a4 + 210a8 - (44/3)a0 - 44a4 - 1540a8 = -(44/3)a0 - 43a4 - 1330a8

# p[3] = -28a8 - 6930*(-(a0+3a4+105a8)/945) = -28a8 + (6930/945)*(a0+3a4+105a8)
# 6930/945 = 7.333... = 22/3
# p[3] = -28a8 + (22/3)*(a0+3a4+105a8) = -28a8 + (22/3)a0 + 22a4 + 770a8 = (22/3)a0 + 22a4 + 742a8

# p[4] = a8 + 1155*(-(a0+3a4+105a8)/945) = a8 - (1155/945)*(a0+3a4+105a8)
# 1155/945 = 1.2222... = 11/9
# p[4] = a8 - (11/9)*(a0+3a4+105a8) = a8 - (11/9)a0 - 11a4 - 125a8 = -(11/9)a0 - 11a4 - 124a8

# p[5] = -66*(-(a0+3a4+105a8)/945) = (66/945)*(a0+3a4+105a8) = (22/315)*(a0+3a4+105a8)
# p[5] = (22/315)a0 + (22/315)*3a4 + (22/315)*105a8 = (22/315)a0 + (66/315)a4 + (2310/315)a8 = (22/315)a0 + (22/105)a4 + (22/3)a8

# p[6] = -(a0+3a4+105a8)/945 = -(1/945)a0 - (3/945)a4 - (105/945)a8 = -(1/945)a0 - (1/315)a4 - (1/9)a8

# So the full mapping from (a0,a4,a8) to p = [p0,...,p6] where p0=0:
# p0 = 0
# p1 = 8*a0 + 18*a4 + 420*a8
# p2 = -(44/3)*a0 - 43*a4 - 1330*a8
# p3 = (22/3)*a0 + 22*a4 + 742*a8
# p4 = -(11/9)*a0 - 11*a4 - 124*a8
# p5 = (22/315)*a0 + (22/105)*a4 + (22/3)*a8
# p6 = -(1/945)*a0 - (1/315)*a4 - (1/9)*a8

# Let me define the matrix M such that p = M @ [a0, a4, a8]
_M = np.array([
    [0.0, 0.0, 0.0],           # p0
    [8.0, 18.0, 420.0],        # p1
    [-44.0/3.0, -43.0, -1330.0],  # p2
    [22.0/3.0, 22.0, 742.0],   # p3
    [-11.0/9.0, -11.0, -124.0], # p4
    [22.0/315.0, 22.0/105.0, 22.0/3.0],  # p5
    [-1.0/945.0, -1.0/315.0, -1.0/9.0]   # p6
])

# Precompute I table slices for efficiency
# For m_k = sum_{i,j} p_i p_j I[k+i+j], we need I up to index 6+6+2=14


def _compute_moments(p):
    """Compute m0, m1, m2 where m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]"""
    I = _I_TABLE
    m = np.zeros(3)
    for k in range(3):
        s = 0.0
        for i in range(7):
            if p[i] == 0:
                continue
            for j in range(7):
                if p[j] == 0:
                    continue
                s += p[i] * p[j] * I[k+i+j]
        m[k] = s
    return m


def _compute_moments_vec(p):
    """Vectorized computation of m0, m1, m2"""
    I = _I_TABLE
    # m_k = p^T @ I_k @ p where I_k[i,j] = I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        # Build the matrix I_k implicitly
        s_p = np.zeros(7)
        for i in range(7):
            s_p[i] = np.dot(p, I[k+i:k+i+7])
        m[k] = np.dot(p, s_p)
    return m


def _bound_and_grad(params):
    """
    Compute uncertainty bound and its analytic gradient.
    params = [a0, a4, a8]
    """
    a0, a4, a8 = params
    # p = M @ [a0, a4, a8]
    p = _M @ np.array([a0, a4, a8])
    
    I = _I_TABLE
    
    # Compute m0, m1, m2 for position space
    m = _compute_moments_vec(p)
    m0, m1, m2 = m[0], m[1], m[2]
    
    # For Fourier: q_k = p_k * (-1)^k
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    # Compute n0, n1, n2 for Fourier space (with 2*pi factor)
    n = _compute_moments_vec(q)
    n0, n1, n2 = 2.0*np.pi*n[0], 2.0*np.pi*n[1], 2.0*np.pi*n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0  # Var(x)
    vF = n2 / n0  # Var(xi)
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient using chain rule
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    # dm_k/da_c = 2 * sum_i (dp_i/da_c) * (sum_j p_j I[k+i+j])
    
    grad = np.zeros(3)
    
    # Precompute s_p and s_q for efficiency
    s_p = np.zeros(7)
    s_q = np.zeros(7)
    for i in range(7):
        s_p[i] = np.dot(p, I[i:i+7])
        s_q[i] = np.dot(q, I[i:i+7])
    
    # Also need shifted versions for m1, m2
    s_p1 = np.zeros(7)
    s_p2 = np.zeros(7)
    s_q1 = np.zeros(7)
    s_q2 = np.zeros(7)
    for i in range(7):
        s_p1[i] = np.dot(p, I[i+1:i+8])
        s_p2[i] = np.dot(p, I[i+2:i+9])
        s_q1[i] = np.dot(q, I[i+1:i+8])
        s_q2[i] = np.dot(q, I[i+2:i+9])
    
    for c in range(3):
        dp = _M[:, c]  # dp/da_c
        dq = dp * signs
        
        dm0 = 2.0 * np.dot(dp, s_p)
        dm1 = 2.0 * np.dot(dp, s_p1)
        dm2 = 2.0 * np.dot(dp, s_p2)
        
        dn0 = 2.0 * 2.0 * np.pi * np.dot(dq, s_q)
        dn1 = 2.0 * 2.0 * np.pi * np.dot(dq, s_q1)
        dn2 = 2.0 * 2.0 * np.pi * np.dot(dq, s_q2)
        
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
    # Diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0005]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-1.8, 0.12, 0.001]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([-2.5, 0.13, 0.0015]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-4.0, 0.2, 0.003]),
    ]
    
    rng = np.random.default_rng(12345)
    # Add random starts with various scales
    for scale in [0.3, 0.8, 1.5, 3.0]:
        for _ in range(10):
            starts.append(rng.normal(0, scale, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 2: Polish with Nelder-Mead
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
