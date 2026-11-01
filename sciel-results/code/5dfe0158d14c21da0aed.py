"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I table and sign array at module level
_I_TABLE = np.zeros(14)
_I_TABLE[0] = np.sqrt(np.pi)
for _n in range(1, 14):
    _I_TABLE[_n] = (2*_n - 1) / 2.0 * _I_TABLE[_n-1]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_moments(p, I):
    """Vectorized computation of m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2"""
    # Build matrix M[k, i, j] = I[k+i+j] for valid indices
    # Use outer product approach
    pp = np.outer(p, p)  # 7x7
    m = np.zeros(3)
    for k in range(3):
        # Sum pp[i,j] * I[k+i+j] over valid (i,j)
        total = 0.0
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    total += pp[i, j] * I[idx]
        m[k] = total
    return m


def _compute_moments_fast(p, I):
    """Even faster: precompute the kernel matrix"""
    # K[i,j] for each k: I[k+i+j]
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    # = p^T K_k p where K_k[i,j] = I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        # Build 7x7 matrix K_k
        K = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    K[i, j] = I[idx]
        m[k] = p @ K @ p
    return m


def _precompute_kernels(I):
    """Precompute the 3 kernel matrices K_0, K_1, K_2"""
    kernels = []
    for k in range(3):
        K = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    K[i, j] = I[idx]
        kernels.append(K)
    return kernels

_KERNELS = _precompute_kernels(_I_TABLE)


def _moments_from_coeffs(p, kernels):
    """Compute moments using precomputed kernels: m_k = p^T K_k p"""
    m = np.zeros(3)
    for k in range(3):
        m[k] = p @ kernels[k] @ p
    return m


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
    
    # Compute moments using precomputed kernels
    m = np.zeros(3)
    for k in range(3):
        m[k] = p @ _KERNELS[k] @ p
    
    # For Fourier transform: Q(xi) coefficients
    q = p * _SIGNS
    
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * (q @ _KERNELS[k] @ q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # da12/da_c
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp/da_c matrix: 7x3
    # Row 0: [1, 3, 105] + 945*da12
    # Row 1: [0, -6, -420] - 7560*da12
    # Row 2: [0, 1, 210] + 13860*da12
    # Row 3: [0, 0, -28] - 6930*da12
    # Row 4: [0, 0, 1] + 1155*da12
    # Row 5: [0, 0, 0] - 66*da12
    # Row 6: [0, 0, 0] + 1*da12
    dp_dac = np.zeros((7, 3))
    dp_dac[0] = np.array([1.0, 3.0, 105.0]) + 945.0*da12
    dp_dac[1] = np.array([0.0, -6.0, -420.0]) - 7560.0*da12
    dp_dac[2] = np.array([0.0, 1.0, 210.0]) + 13860.0*da12
    dp_dac[3] = np.array([0.0, 0.0, -28.0]) - 6930.0*da12
    dp_dac[4] = np.array([0.0, 0.0, 1.0]) + 1155.0*da12
    dp_dac[5] = np.array([0.0, 0.0, 0.0]) - 66.0*da12
    dp_dac[6] = np.array([0.0, 0.0, 0.0]) + 1.0*da12
    
    # dq/da_c = dp/da_c * signs (element-wise per row)
    dq_dac = dp_dac * _SIGNS[:, None]
    
    grad = np.zeros(3)
    for c in range(3):
        dp_c = dp_dac[:, c]
        dq_c = dq_dac[:, c]
        
        # dm_k/da_c = 2 * dp_c^T K_k p
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * (dp_c @ _KERNELS[k] @ p)
        
        # dn_k/da_c = 2*pi * 2 * dq_c^T K_k q
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 4.0 * np.pi * (dq_c @ _KERNELS[k] @ q)
        
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
    # Structured multi-start: seed + grid + random
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.05, 0.0005]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-2.5, 0.12, 0.0012]),
        np.array([-1.8, 0.07, 0.0007]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.1, 0.001]),
        np.array([-0.7, 0.025, 0.00025]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.4, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15,
                                             'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 25:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-18,
                                   'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16,
                                             'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
