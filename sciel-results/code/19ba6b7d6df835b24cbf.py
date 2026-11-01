"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # Safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute the derivative coefficient matrix
# dp_k/da_c for c=0,1,2 (a0, a4, a8)
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
_dp_da = np.zeros((7, 3))
_dp_da[0, 0] = 1.0 + 945.0 * _da12[0]
_dp_da[0, 1] = 3.0 + 945.0 * _da12[1]
_dp_da[0, 2] = 105.0 + 945.0 * _da12[2]
_dp_da[1, 0] = 0.0 - 7560.0 * _da12[0]
_dp_da[1, 1] = -6.0 - 7560.0 * _da12[1]
_dp_da[1, 2] = -420.0 - 7560.0 * _da12[2]
_dp_da[2, 0] = 0.0 + 13860.0 * _da12[0]
_dp_da[2, 1] = 1.0 + 13860.0 * _da12[1]
_dp_da[2, 2] = 210.0 + 13860.0 * _da12[2]
_dp_da[3, 0] = 0.0 - 6930.0 * _da12[0]
_dp_da[3, 1] = 0.0 - 6930.0 * _da12[1]
_dp_da[3, 2] = -28.0 - 6930.0 * _da12[2]
_dp_da[4, 0] = 0.0 + 1155.0 * _da12[0]
_dp_da[4, 1] = 0.0 + 1155.0 * _da12[1]
_dp_da[4, 2] = 1.0 + 1155.0 * _da12[2]
_dp_da[5, 0] = 0.0 - 66.0 * _da12[0]
_dp_da[5, 1] = 0.0 - 66.0 * _da12[1]
_dp_da[5, 2] = 0.0 - 66.0 * _da12[2]
_dp_da[6, 0] = 0.0 + 1.0 * _da12[0]
_dp_da[6, 1] = 0.0 + 1.0 * _da12[1]
_dp_da[6, 2] = 0.0 + 1.0 * _da12[2]

# Precompute (-1)^k for q_k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute I matrices for moment calculations
# I_mat_k[i,j] = I[k+i+j] for k=0,1,2
_I_mat = [np.zeros((7, 7)) for _ in range(3)]
for k in range(3):
    for i in range(7):
        for j in range(7):
            _I_mat[k][i, j] = _I[k + i + j]

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
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    # q_k = p_k * (-1)^k
    q = p * _signs
    
    # Vectorized moment calculations
    # m_k = p^T I_mat_k p
    m0 = p @ _I_mat[0] @ p
    m1 = p @ _I_mat[1] @ p
    m2 = p @ _I_mat[2] @ p
    
    # n_k = 2*pi * q^T I_mat_k q
    n0 = 2.0 * np.pi * (q @ _I_mat[0] @ q)
    n1 = 2.0 * np.pi * (q @ _I_mat[1] @ q)
    n2 = 2.0 * np.pi * (q @ _I_mat[2] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[:, c]
        dq = dp * _signs
        
        # dm_k/da_c = 2 * dp^T I_mat_k p
        dm0 = 2.0 * dp @ _I_mat[0] @ p
        dm1 = 2.0 * dp @ _I_mat[1] @ p
        dm2 = 2.0 * dp @ _I_mat[2] @ p
        
        # dn_k/da_c = 2 * 2*pi * dq^T I_mat_k q
        dn0 = 4.0 * np.pi * dq @ _I_mat[0] @ q
        dn1 = 4.0 * np.pi * dq @ _I_mat[1] @ q
        dn2 = 4.0 * np.pi * dq @ _I_mat[2] @ q
        
        dvm2 = (dm1*m0 - m1*dm0) / (m0*m0)
        dvn2 = (dn1*n0 - n1*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points - expanded for better coverage
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0005]),
        np.array([-2.5, 0.07, 0.0007]),
        np.array([-1.8, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    # Add more random starts with varying scales
    for i in range(40):
        scale = 0.2 + 0.4 * i / 40
        starts.append(rng.normal(0, scale, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish best with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
