"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # Safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    # Need indices up to 2 + 6 + 6 = 14, so size must be at least 15
    I = np.zeros(15)
    sp = np.sqrt(np.pi)
    for n in range(15):
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
    # Structural Change: Use Differential Evolution for global search
    # This avoids the need for manually tuned multi-starts and handles
    # the non-convexity of the uncertainty bound better.
    
    # Define bounds for the coefficients.
    # Based on the seed and typical scales:
    # a0 is negative, around -1.
    # a4 is positive, small.
    # a8 is positive, very small.
    bounds = [
        (-10.0, 10.0),   # a0
        (-5.0, 5.0),     # a4
        (-1.0, 1.0)      # a8
    ]
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: Global Search with Differential Evolution
    if time.time() < _DEADLINE - 30:
        try:
            # DE requires a function that returns only the value
            # maxiter is set high, but we rely on the deadline check via callback or just letting it run if fast
            # For 3D problem, DE is fast.
            res_de = minimize(_objective, np.array([-1.0, 0.0, 0.0]), method='differential_evolution',
                              bounds=bounds,
                              maxiter=200,
                              tol=1e-8,
                              seed=42,
                              polish=False) # We will polish manually
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # Fallback to seed if DE failed or didn't find anything better
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
        best_val = _objective(best_params)
    
    # Phase 2: Local Polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res_lbfgs = minimize(_objective_grad, best_params, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res_lbfgs.fun < best_val and np.isfinite(res_lbfgs.fun):
                best_val = res_lbfgs.fun
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
            
    # Phase 3: Final tight polish
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            res_final = minimize(_objective_grad, best_params, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res_final.fun < best_val and np.isfinite(res_final.fun):
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
