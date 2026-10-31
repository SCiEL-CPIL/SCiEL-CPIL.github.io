"""Optimized uncertainty bound via constrained eigenvalue problem + multi-start Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill

def _gaussian_moment(m):
    """Compute G[m] = integral x^{2m} exp(-x^2) dx."""
    if m < 0:
        return 0.0
    val = np.sqrt(np.pi)
    for i in range(1, m + 1):
        val *= (2 * i - 1) / 2.0
    return val

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound for given coefficients.
    Returns the bound value.
    """
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # Constraint: P(0) = 0 => a0 + 3a4 + 105a8 + 945a12 = 0
    # => a12 = -(a0 + 3a4 + 105a8) / 945
    
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^6 p_k x^{2k}
    # H0 = 1
    # H4 = 3 - 6x^2 + x^4
    # H8 = 105 - 420x^2 + 210x^4 - 28x^6 + x^8
    # H12 = 945 - 7560x^2 + 13860x^4 - 6930x^6 + 1155x^8 - 66x^10 + x^12
    
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12  # This is 0 by construction
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = 1.0*a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = 1.0*a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = 1.0*a12
    
    # M_0 = <P^2, e^{-x^2}> = sum_{j,k} p_j p_k G[j+k]
    # M_2 = <P^2 x^2, e^{-x^2}> = sum_{j,k} p_j p_k G[j+k+1]
    
    M_0 = 0.0
    M_2 = 0.0
    
    for j in range(7):
        if p[j] == 0: continue
        for k in range(7):
            val = p[j] * p[k]
            M_0 += val * _gaussian_moment(j + k)
            M_2 += val * _gaussian_moment(j + k + 1)
            
    if M_0 <= 0:
        return np.inf
        
    # Q(x) = -P'(x)
    # P'(x) = sum_{k=1}^6 2k p_k x^{2k-1}
    # Q(x) = - sum_{k=1}^6 2k p_k x^{2k-1}
    # Let Q(x) = sum_{m=0}^5 q_m x^{2m+1}
    # q_0 = -2 p_1
    # q_1 = -4 p_2
    # ...
    # q_m = -2(m+1) p_{m+1}
    
    q = np.zeros(6)
    for m in range(6):
        q[m] = -2.0 * (m + 1) * p[m + 1]
        
    # N_2 = <Q^2 x^2, e^{-x^2}> = <(P')^2 x^2, e^{-x^2}>
    # Q(x) = sum_{m=0}^5 q_m x^{2m+1}
    # Q^2 x^2 = (sum_{m} q_m x^{2m+1})^2 x^2 = sum_{m,n} q_m q_n x^{2m+2n+4}
    # Integral of x^{2m+2n+4} e^{-x^2} is G[m+n+2]
    
    N_2 = 0.0
    for m in range(6):
        if q[m] == 0: continue
        for n in range(6):
            N_2 += q[m] * q[n] * _gaussian_moment(m + n + 2)
            
    if N_2 < 0:
        return np.inf
        
    bound = np.sqrt(M_2 * N_2) / M_0
    return bound

def _compute_bound_grad(a0, a4, a8):
    """
    Compute the uncertainty bound and its gradient using finite differences.
    This is slower but robust.
    """
    val = _compute_bound(a0, a4, a8)
    if not np.isfinite(val):
        return val, np.array([0.0, 0.0, 0.0])
        
    h = 1e-6
    x = np.array([a0, a4, a8])
    grad = np.zeros(3)
    for i in range(3):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        val_plus = _compute_bound(*x_plus)
        val_minus = _compute_bound(*x_minus)
        grad[i] = (val_plus - val_minus) / (2 * h)
        
    return val, grad

def _objective(x):
    if time.time() > _DEADLINE:
        return np.inf
    try:
        return _compute_bound(x[0], x[1], x[2])
    except:
        return np.inf

def _objective_grad(x):
    if time.time() > _DEADLINE:
        return np.zeros(3)
    try:
        _, grad = _compute_bound_grad(x[0], x[1], x[2])
        return grad
    except:
        return np.zeros(3)

def construct():
    from scipy.optimize import minimize, dual_annealing
    
    # Start from the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _compute_bound(x0[0], x0[1], x0[2])
    
    # 1. Global Search: Dual Annealing
    # We use a wide box to ensure we cover the potential landscape
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    try:
        res_da = dual_annealing(
            _objective,
            bounds,
            maxiter=100,
            restarts=30,
            maxfun=5000,
            initial_temp=5230.0,
            visit='2-tent',
            seed=42
        )
        if res_da.fun < best_val:
            best_val = res_da.fun
            best_x = res_da.x.copy()
    except Exception as e:
        pass
        
    # 2. Local Refinement: L-BFGS-B with finite difference gradient
    # This is more robust than analytic gradients which can have subtle errors
    try:
        res_lbfgs = minimize(
            _objective,
            best_x,
            jac=_objective_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if res_lbfgs.fun < best_val:
            best_val = res_lbfgs.fun
            best_x = res_lbfgs.x.copy()
    except Exception as e:
        pass
        
    # 3. Final Polish: Nelder-Mead for high precision
    try:
        res_nm = minimize(
            _objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_x = res_nm.x.copy()
    except Exception as e:
        pass
        
    return best_x
# EVOLVE-BLOCK-END
