import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

START_TIME = time.time()
TIME_LIMIT = 205.0  # Safety margin before 210s

# Precompute Hermite polynomial coefficients in power basis [x^0, x^2, ..., x^12]
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 226800x^2 + 945

# Coefficients for P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# We express P(x) = sum_{k=0}^6 p_k x^{2k}
# The constraint g(0)=0 implies P(0)=0 => c0 + 3c4 + 105c8 + 945c12 = 0
# c12 = -(c0 + 3c4 + 105c8) / 945

# Precompute integrals I_n = integral_{-inf}^{inf} x^{2n} e^{-x^2} dx
# I_n = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_integrals():
    I = np.zeros(28) # Need up to x^26 or x^28 for safety
    sp = np.sqrt(np.pi)
    for n in range(28):
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I_n = (2n-1)/2 * I_{n-1}
            I[n] = (2*n - 1) / 2.0 * I[n-1]
    return I

_I = _precompute_integrals()

# Derivatives of p_k with respect to c0, c4, c8
# p0 = c0 + 3c4 + 105c8 + 945c12
# p1 = -6c4 - 420c8 - 7560c12
# p2 = c4 + 210c8 + 13860c12
# p3 = -28c8 - 6930c12
# p4 = c8 + 1155c12
# p5 = -66c12
# p6 = c12

# c12 = -(c0 + 3c4 + 105c8)/945
# dc12/dc0 = -1/945, dc12/dc4 = -3/945, dc12/dc8 = -105/945

# dp_k/dc_j matrix (7x3)
# Row k, Col j
_DP_DC = np.zeros((7, 3))
# k=0
_DP_DC[0, 0] = 1.0 + 945.0 * (-1.0/945.0)   # 1 - 1 = 0
_DP_DC[0, 1] = 3.0 + 945.0 * (-3.0/945.0)   # 3 - 3 = 0
_DP_DC[0, 2] = 105.0 + 945.0 * (-105.0/945.0) # 105 - 105 = 0
# Wait, p0 is the constant term.
# p0 = c0 + 3c4 + 105c8 + 945c12
# Since c12 is defined to make p0=0, the effective p0 in the optimized function is always 0?
# No, the function is g(x) = P(x) exp(-x^2/2).
# P(x) is the polynomial.
# The constraint is P(0)=0.
# So we substitute c12 into P(x).
# The resulting P(x) has p0=0 identically.
# So we only need to vary c0, c4, c8, and p0 is fixed at 0.
# Let's re-verify the coefficients p_k for the substituted polynomial.

# c12 = - (c0 + 3c4 + 105c8) / 945
# Let S = c0 + 3c4 + 105c8
# c12 = -S/945

# p0 = c0 + 3c4 + 105c8 + 945c12 = S - S = 0.
# p1 = -6c4 - 420c8 - 7560c12 = -6c4 - 420c8 + 7560/945 * S = -6c4 - 420c8 + 8 * S
#    = -6c4 - 420c8 + 8(c0 + 3c4 + 105c8)
#    = 8c0 + (-6+24)c4 + (-420+840)c8
#    = 8c0 + 18c4 + 420c8
# p2 = c4 + 210c8 + 13860c12 = c4 + 210c8 - 13860/945 * S
#    13860/945 = 14.666... = 44/3
#    = c4 + 210c8 - (44/3)(c0 + 3c4 + 105c8)
#    = -(44/3)c0 + (1 - 44)c4 + (210 - 44*105/3)c8
#    44*105/3 = 44*35 = 1540
#    = -(44/3)c0 - 43c4 + (210 - 1540)c8
#    = -(44/3)c0 - 43c4 - 1330c8
# p3 = -28c8 - 6930c12 = -28c8 + 6930/945 * S
#    6930/945 = 7.333... = 22/3
#    = -28c8 + (22/3)(c0 + 3c4 + 105c8)
#    = (22/3)c0 + 22c4 + (22*105/3 - 28)c8
#    22*105/3 = 22*35 = 770
#    = (22/3)c0 + 22c4 + (770 - 28)c8
#    = (22/3)c0 + 22c4 + 742c8
# p4 = c8 + 1155c12 = c8 - 1155/945 * S
#    1155/945 = 77/63 = 11/9
#    = c8 - (11/9)(c0 + 3c4 + 105c8)
#    = -(11/9)c0 - 11c4 + (1 - 11*105/9)c8
#    11*105/9 = 11*35/3 = 385/3 = 128.333
#    = -(11/9)c0 - 11c4 + (1 - 385/3)c8
#    = -(11/9)c0 - 11c4 - 382/3 c8
# p5 = -66c12 = 66/945 S = 22/315 S
#    = (22/315)c0 + (66/315)c4 + (2310/315)c8
#    66/315 = 22/105
#    2310/315 = 22/3
#    = (22/315)c0 + (22/105)c4 + (22/3)c8
# p6 = c12 = -1/945 S
#    = -1/945 c0 - 3/945 c4 - 105/945 c8
#    = -1/945 c0 - 1/315 c4 - 1/9 c8

# Let's verify these derivatives numerically or just use the symbolic substitution in code.
# It is safer to compute p_k(c) directly in the function and use automatic differentiation or finite differences for the gradient if analytic is too complex?
# No, the inspiration code has the analytic gradient. I will replicate that logic carefully.

def _get_p(c0, c4, c8):
    """Returns coefficients p[0..6] of P(x) given c0, c4, c8."""
    S = c0 + 3.0*c4 + 105.0*c8
    c12 = -S / 945.0
    
    # p0 is 0
    p0 = 0.0
    p1 = -6.0*c4 - 420.0*c8 - 7560.0*c12
    p2 = c4 + 210.0*c8 + 13860.0*c12
    p3 = -28.0*c8 - 6930.0*c12
    p4 = c8 + 1155.0*c12
    p5 = -66.0*c12
    p6 = c12
    return np.array([p0, p1, p2, p3, p4, p5, p6])

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    """
    c0, c4, c8 = params[0], params[1], params[2]
    p = _get_p(c0, c4, c8)
    
    # Fourier transform coefficients q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
        
    # Moments m_k = integral x^k |g|^2 dx = sum_{i,j} p_i p_j I[k+i+j]
    # We need m0, m2 for variance <x^2> = m2/m0
    # And n_k = integral xi^k |G|^2 dx = 2*pi sum_{i,j} q_i q_j I[k+i+j]
    # We need n0, n2 for variance <xi^2> = n2/n0
    
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            qq = q[i] * q[j]
            
            # m0: k=0, idx = i+j
            idx = i + j
            if idx < len(_I):
                m0 += pp * _I[idx]
                
            # m2: k=2, idx = i+j+2
            idx = i + j + 2
            if idx < len(_I):
                m2 += pp * _I[idx]
                
            # n0: k=0, idx = i+j
            idx = i + j
            if idx < len(_I):
                n0 += qq * _I[idx]
                
            # n2: k=2, idx = i+j+2
            idx = i + j + 2
            if idx < len(_I):
                n2 += qq * _I[idx]
                
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # bound = (vf * vF)^0.5
    # dbound/dc = 0.5 * (vf*vF)^-0.5 * (vF * dvf/dc + vf * dvF/dc)
    # dvf/dc = (dm2/dc * m0 - m2 * dm0/dc) / m0^2
    # dvF/dc = (dn2/dc * n0 - n2 * dn0/dc) / n0^2
    
    grad = np.zeros(3)
    
    # Precompute sums for gradient
    # dm0/dc_j = 2 * sum_i dp_i/dc_j * sum_j p_j I[i+j]
    # dm2/dc_j = 2 * sum_i dp_i/dc_j * sum_j p_j I[i+j+2]
    # Same for n with q and dq/dc
    
    # We need dp/dc matrix
    # From previous derivation:
    # p1 = 8c0 + 18c4 + 420c8
    # p2 = -(44/3)c0 - 43c4 - 1330c8
    # p3 = (22/3)c0 + 22c4 + 742c8
    # p4 = -(11/9)c0 - 11c4 - 382/3 c8
    # p5 = (22/315)c0 + (22/105)c4 + (22/3)c8
    # p6 = -1/945 c0 - 1/315 c4 - 1/9 c8
    
    dp_dc = np.zeros((7, 3))
    # Row 0 is 0
    dp_dc[1, :] = [8.0, 18.0, 420.0]
    dp_dc[2, :] = [-44.0/3.0, -43.0, -1330.0]
    dp_dc[3, :] = [22.0/3.0, 22.0, 742.0]
    dp_dc[4, :] = [-11.0/9.0, -11.0, -382.0/3.0]
    dp_dc[5, :] = [22.0/315.0, 22.0/105.0, 22.0/3.0]
    dp_dc[6, :] = [-1.0/945.0, -1.0/315.0, -1.0/9.0]
    
    dq_dc = np.zeros((7, 3))
    for k in range(7):
        dq_dc[k, :] = dp_dc[k, :] * ((-1.0)**k)
        
    # Precompute S_p[i] = sum_j p_j I[i+j] and S_p2[i] = sum_j p_j I[i+j+2]
    S_p = np.zeros(7)
    S_p2 = np.zeros(7)
    S_q = np.zeros(7)
    S_q2 = np.zeros(7)
    
    for i in range(7):
        for j in range(7):
            idx0 = i + j
            idx2 = i + j + 2
            if idx0 < len(_I):
                S_p[i] += p[j] * _I[idx0]
                S_q[i] += q[j] * _I[idx0]
            if idx2 < len(_I):
                S_p2[i] += p[j] * _I[idx2]
                S_q2[i] += q[j] * _I[idx2]
                
    for c_idx in range(3):
        # dm0/dc = 2 * sum_i dp_i/dc * S_p[i]
        dm0_dc = 2.0 * np.dot(dp_dc[:, c_idx], S_p)
        dm2_dc = 2.0 * np.dot(dp_dc[:, c_idx], S_p2)
        
        # dn0/dc = 2 * 2*pi * sum_i dq_i/dc * S_q[i]
        dn0_dc = 4.0 * np.pi * np.dot(dq_dc[:, c_idx], S_q)
        dn2_dc = 4.0 * np.pi * np.dot(dq_dc[:, c_idx], S_q2)
        
        dvf_dc = (dm2_dc * m0 - m2 * dm0_dc) / (m0 * m0)
        dvF_dc = (dn2_dc * n0 - n2 * dn0_dc) / (n0 * n0)
        
        grad[c_idx] = (vF * dvf_dc + vf * dvF_dc) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Seed initial guess
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    # Phase 1: Differential evolution for global exploration
    # DE is gradient-free, so it's robust. We give it a decent chunk of time.
    if time.time() < TIME_LIMIT - 80:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False # We will polish separately
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with analytic gradient
    # This is very fast.
    num_starts = 30
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 20:
            break
        
        # Generate diverse initial guesses
        if i == 0:
            x0 = best_coeffs.copy()
        elif i == 1:
            x0 = seed.copy()
        elif i == 2:
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, 0.01])
        elif i == 4:
            x0 = np.array([1.0, -0.1, -0.01])
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        try:
            res = minimize(_bound_and_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Targeted refinement near best solution
    for i in range(10):
        if time.time() > TIME_LIMIT - 10:
            break
        
        # Perturb the best solution with decreasing scale
        scale = 0.1 * (0.5 ** (i // 5))
        x0 = best_coeffs + np.random.normal(0, scale, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_bound_and_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Final high-precision polish
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(_bound_and_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
