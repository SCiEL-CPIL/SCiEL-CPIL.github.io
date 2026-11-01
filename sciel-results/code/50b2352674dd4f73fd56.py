# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute constants
_I_TABLE = np.zeros(14)
def _init_I_table():
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        _I_TABLE[n] = np.exp(log_val)

_init_I_table()

# Precompute S_k matrices for m_k = p^T S_k p
# S_k[i,j] = I[k+i+j]
_S_M = [np.zeros((7,7)) for _ in range(3)]
_S_N = [np.zeros((7,7)) for _ in range(3)]
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                val = _I_TABLE[idx]
                _S_M[k][i,j] = val
                _S_N[k][i,j] = 2.0 * np.pi * val * ((-1.0)**(i+j))

# Precompute base vectors for dp/da_c
_H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
_H12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

def _bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    q = p * _SIGN
    
    m0 = p @ (_S_M[0] @ p)
    m2 = p @ (_S_M[1] @ p)
    
    n0 = q @ (_S_N[0] @ q)
    n2 = q @ (_S_N[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    Sp0 = _S_M[0] @ p
    Sp1 = _S_M[1] @ p
    Sq0 = _S_N[0] @ q
    Sq1 = _S_N[1] @ q
    
    grad = np.zeros(3)
    bases = [_H0, _H4, _H8]
    for c in range(3):
        dp = bases[c] + _DA12[c] * _H12
        dq = dp * _SIGN
        
        dm0 = 2.0 * (Sp0 @ dp)
        dm2 = 2.0 * (Sp1 @ dp)
        dn0 = 2.0 * (Sq0 @ dq)
        dn2 = 2.0 * (Sq1 @ dq)
        
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
    # Targeted starts near known good solution
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.00031]),
        np.array([-1.12, 0.04, 0.0003]),
        np.array([-1.14, 0.04, 0.0003]),
        np.array([-1.13, 0.03, 0.0003]),
        np.array([-1.13, 0.05, 0.0003]),
        np.array([-1.13, 0.04, 0.0002]),
        np.array([-1.13, 0.04, 0.0004]),
    ]
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-precise pass
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
