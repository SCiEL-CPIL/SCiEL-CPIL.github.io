"""Analytic uncertainty bound with precomputed integrals + vectorized gradient + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 12 (for x^{24} terms)
_I_CACHE = None

def _get_I():
    """Return precomputed array I[n] for n=0..13."""
    global _I_CACHE
    if _I_CACHE is not None:
        return _I_CACHE
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        # I[n] = I[n-1] * (2n-1) / (2n)
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    _I_CACHE = I
    return I


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    Returns (bound, grad) where bound = sqrt(vf * vF)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    I = _get_I()
    
    # Polynomial coefficients p_k for P(x) = sum p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # |g|^2 = P(x)^2 * e^{-x^2}
    # m_k = integral x^{2k} |g|^2 dx = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    # Vectorize: m_k = p @ (I_shifted[k]) @ p where I_shifted[k][i,j] = I[k+i+j]
    # Build matrix M_k where M_k[i,j] = I[k+i+j]
    
    # Precompute the 7x7 matrices for k=0,1,2
    # M_k[i,j] = I[k+i+j]
    M0 = np.array([[I[i+j] for j in range(7)] for i in range(7)])
    M1 = np.array([[I[1+i+j] for j in range(7)] for i in range(7)])
    M2 = np.array([[I[2+i+j] for j in range(7)] for i in range(7)])
    
    m0 = p @ M0 @ p
    m1 = p @ M1 @ p  # this is m2 (integral x^2 |g|^2)
    m2 = p @ M2 @ p  # this is m4 (integral x^4 |g|^2)
    
    # Fourier: Q(xi) = sum q_k xi^{2k}, q_k = p_k * (-1)^k
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    # |G|^2 = 2*pi * Q(xi)^2 * e^{-xi^2}
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    two_pi = 2.0 * np.pi
    n0 = two_pi * (q @ M0 @ q)
    n1 = two_pi * (q @ M1 @ q)
    n2 = two_pi * (q @ M2 @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # da12/da_c for c=0,1,2 (corresponding to a0, a4, a8)
    da12_da = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c matrix: shape (3, 7)
    # dp0/da_c = [1,3,105][c] + 945*da12_da[c]
    # dp1/da_c = [0,-6,-420][c] - 7560*da12_da[c]
    # dp2/da_c = [0,1,210][c] + 13860*da12_da[c]
    # dp3/da_c = [0,0,-28][c] - 6930*da12_da[c]
    # dp4/da_c = [0,0,1][c] + 1155*da12_da[c]
    # dp5/da_c = [0,0,0][c] - 66*da12_da[c]
    # dp6/da_c = [0,0,0][c] + 1*da12_da[c]
    
    dp_da = np.zeros((3, 7))
    dp_da[0, 0] = 1.0 + 945.0 * da12_da[0]
    dp_da[1, 0] = 3.0 + 945.0 * da12_da[1]
    dp_da[2, 0] = 105.0 + 945.0 * da12_da[2]
    
    dp_da[0, 1] = 0.0 - 7560.0 * da12_da[0]
    dp_da[1, 1] = -6.0 - 7560.0 * da12_da[1]
    dp_da[2, 1] = -420.0 - 7560.0 * da12_da[2]
    
    dp_da[0, 2] = 0.0 + 13860.0 * da12_da[0]
    dp_da[1, 2] = 1.0 + 13860.0 * da12_da[1]
    dp_da[2, 2] = 210.0 + 13860.0 * da12_da[2]
    
    dp_da[0, 3] = 0.0 - 6930.0 * da12_da[0]
    dp_da[1, 3] = 0.0 - 6930.0 * da12_da[1]
    dp_da[2, 3] = -28.0 - 6930.0 * da12_da[2]
    
    dp_da[0, 4] = 0.0 + 1155.0 * da12_da[0]
    dp_da[1, 4] = 0.0 + 1155.0 * da12_da[1]
    dp_da[2, 4] = 1.0 + 1155.0 * da12_da[2]
    
    dp_da[0, 5] = 0.0 - 66.0 * da12_da[0]
    dp_da[1, 5] = 0.0 - 66.0 * da12_da[1]
    dp_da[2, 5] = 0.0 - 66.0 * da12_da[2]
    
    dp_da[0, 6] = 1.0 * da12_da[0]
    dp_da[1, 6] = 1.0 * da12_da[1]
    dp_da[2, 6] = 1.0 * da12_da[2]
    
    # For each parameter c, dm_k/da_c = 2 * dp_c^T @ M_k @ p
    # where dp_c is the vector dp_k/da_c
    
    # dm/da_c for k=0,1,2: shape (3, 3)
    # dm[cc, k] = 2 * dp_da[cc] @ M_k @ p
    dm_da = np.zeros((3, 3))
    for cc in range(3):
        dm_da[cc, 0] = 2.0 * (dp_da[cc] @ M0 @ p)
        dm_da[cc, 1] = 2.0 * (dp_da[cc] @ M1 @ p)
        dm_da[cc, 2] = 2.0 * (dp_da[cc] @ M2 @ p)
    
    # dq/da_c = dp_da * signs (element-wise per k)
    dq_da = dp_da * signs[np.newaxis, :]
    
    # dn/da_c for k=0,1,2: shape (3, 3)
    dn_da = np.zeros((3, 3))
    for cc in range(3):
        dn_da[cc, 0] = 2.0 * two_pi * (dq_da[cc] @ M0 @ q)
        dn_da[cc, 1] = 2.0 * two_pi * (dq_da[cc] @ M1 @ q)
        dn_da[cc, 2] = 2.0 * two_pi * (dq_da[cc] @ M2 @ q)
    
    # d(vf)/da_c = (dm1/da_c * m0 - m1 * dm0/da_c) / m0^2
    # d(vF)/da_c = (dn1/da_c * n0 - n1 * dn0/da_c) / n0^2
    # d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2 * bound)
    
    m0_sq = m0 * m0
    n0_sq = n0 * n0
    
    dvm = (dm_da[:, 1] * m0 - m1 * dm_da[:, 0]) / m0_sq
    dvn = (dn_da[:, 1] * n0 - n1 * dn_da[:, 0]) / n0_sq
    
    grad = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points with diverse initialization
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0005]),
        np.array([-1.3, 0.03, 0.0004]),
        np.array([-0.7, 0.07, 0.0006]),
        np.array([-1.6, 0.04, 0.0003]),
    ]
    
    rng = np.random.default_rng(42)
    # Add random starts with varying scales
    for i in range(30):
        scale = 0.3 + 0.1 * i
        starts.append(rng.normal(0, scale, 3))
    
    # Sort by initial objective to prioritize promising starts
    start_objs = []
    for s in starts:
        try:
            obj, _ = _bound_and_grad(s)
            if np.isfinite(obj):
                start_objs.append((obj, s))
        except Exception:
            pass
    start_objs.sort(key=lambda x: x[0])
    
    best_val = np.inf
    best_params = None
    
    for obj, start in start_objs:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B at very tight tolerance
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
