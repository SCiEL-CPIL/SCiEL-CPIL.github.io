"""Optimized uncertainty bound minimizer using Global Differential Evolution and Local Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195.0  # Safety margin before 210s kill

# Precompute integration weights I_n = integral x^{2n} e^{-x^2} dx
# I_n = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
# Recurrence: I_n = (2n-1)/4 * I_{n-1}
# We need n up to 13 (max index in variance calc is 1+6+6=13)
_N_MAX = 14
_I = np.zeros(_N_MAX)
_I[0] = np.sqrt(np.pi)
for n in range(1, _N_MAX):
    _I[n] = (2.0 * n - 1.0) / 4.0 * _I[n - 1]

def _compute_bound_and_grad(params):
    """
    Computes the uncertainty bound and its gradient for given a0, a4, a8.
    a12 is determined by g(0)=0.
    """
    a0, a4, a8 = params
    # a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^6 p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p = np.zeros(7)
    # p0 (x^0)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    # p1 (x^2)
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    # p2 (x^4)
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    # p3 (x^6)
    p[3] = -28.0*a8 - 6930.0*a12
    # p4 (x^8)
    p[4] = a8 + 1155.0*a12
    # p5 (x^10)
    p[5] = -66.0*a12
    # p6 (x^12)
    p[6] = a12
    
    # Moments of |g|^2: m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx
    # m_0 = sum_{i,j} p_i p_j I_{i+j}
    # m_2 = sum_{i,j} p_i p_j I_{1+i+j}
    
    P_mat = np.outer(p, p)
    idx = np.add.outer(np.arange(7), np.arange(7))
    
    m0 = np.sum(P_mat * _I[idx])
    m2 = np.sum(P_mat * _I[1 + idx])
    
    # Fourier side
    # q_k = p_k * (-1)^k
    q = p * ((-1.0) ** np.arange(7))
    Q_mat = np.outer(q, q)
    
    n0 = 2.0 * np.pi * np.sum(Q_mat * _I[idx])
    n2 = 2.0 * np.pi * np.sum(Q_mat * _I[1 + idx])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vf * vF)
    
    # Gradient Calculation
    da12_dac = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    Dp_da = np.zeros((3, 7))
    # Row 0 (a0)
    Dp_da[0, 0] = 1.0 + 945.0 * da12_dac[0]
    Dp_da[0, 1] = -7560.0 * da12_dac[0]
    Dp_da[0, 2] = 13860.0 * da12_dac[0]
    Dp_da[0, 3] = -6930.0 * da12_dac[0]
    Dp_da[0, 4] = 1155.0 * da12_dac[0]
    Dp_da[0, 5] = -66.0 * da12_dac[0]
    Dp_da[0, 6] = 1.0 * da12_dac[0]
    
    # Row 1 (a4)
    Dp_da[1, 0] = 3.0 + 945.0 * da12_dac[1]
    Dp_da[1, 1] = -6.0 - 7560.0 * da12_dac[1]
    Dp_da[1, 2] = 1.0 + 13860.0 * da12_dac[1]
    Dp_da[1, 3] = -6930.0 * da12_dac[1]
    Dp_da[1, 4] = 1155.0 * da12_dac[1]
    Dp_da[1, 5] = -66.0 * da12_dac[1]
    Dp_da[1, 6] = 1.0 * da12_dac[1]
    
    # Row 2 (a8)
    Dp_da[2, 0] = 105.0 + 945.0 * da12_dac[2]
    Dp_da[2, 1] = -420.0 - 7560.0 * da12_dac[2]
    Dp_da[2, 2] = 210.0 + 13860.0 * da12_dac[2]
    Dp_da[2, 3] = -28.0 - 6930.0 * da12_dac[2]
    Dp_da[2, 4] = 1.0 + 1155.0 * da12_dac[2]
    Dp_da[2, 5] = -66.0 * da12_dac[2]
    Dp_da[2, 6] = 1.0 * da12_dac[2]
    
    Dq_da = Dp_da * ((-1.0) ** np.arange(7))[np.newaxis, :]
    
    S_p = np.zeros(7)
    for i in range(7):
        S_p[i] = np.dot(p, _I[i + np.arange(7)])
    dm0_da = 2.0 * (Dp_da * S_p[np.newaxis, :]).sum(axis=1)
    
    S_p2 = np.zeros(7)
    for i in range(7):
        S_p2[i] = np.dot(p, _I[1 + i + np.arange(7)])
    dm2_da = 2.0 * (Dp_da * S_p2[np.newaxis, :]).sum(axis=1)
    
    S_q = np.zeros(7)
    for i in range(7):
        S_q[i] = np.dot(q, _I[i + np.arange(7)])
    dn0_da = 2.0 * 2.0 * np.pi * (Dq_da * S_q[np.newaxis, :]).sum(axis=1)
    
    S_q2 = np.zeros(7)
    for i in range(7):
        S_q2[i] = np.dot(q, _I[1 + i + np.arange(7)])
    dn2_da = 2.0 * 2.0 * np.pi * (Dq_da * S_q2[np.newaxis, :]).sum(axis=1)
    
    d_vf = (dm2_da * m0 - m2 * dm0_da) / (m0 * m0)
    d_vF = (dn2_da * n0 - n2 * dn0_da) / (n0 * n0)
    
    grad = (vF * d_vf + vf * d_vF) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    # Initial guess based on previous runs, but we will use global search
    best_params = np.array([-1.13, 0.04, 0.000308642])
    best_val, _ = _compute_bound_and_grad(best_params)
    
    # Define bounds for differential evolution
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)]
    
    # Global search
    if time.time() < _DEADLINE - 30.0:
        try:
            de_res = differential_evolution(
                _objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-8, 
                seed=42,
                polish=False
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x
        except Exception:
            pass
    
    # Local polish with L-BFGS-B
    if time.time() < _DEADLINE - 10.0:
        try:
            res = minimize(
                _objective_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    return np.array(best_params)
# EVOLVE-BLOCK-END
