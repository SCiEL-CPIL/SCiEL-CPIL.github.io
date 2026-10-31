"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi)
_I = np.zeros(14)
for _n in range(14):
    if _n == 0:
        _I[0] = np.sqrt(np.pi)
    else:
        _I[_n] = (2*_n - 1) / (2.0 * _n) * _I[_n - 1]

# Precompute the coefficient mapping matrices
# p_k as function of (a0, a4, a8, a12)
# p0 = a0 + 3*a4 + 105*a8 + 945*a12
# p1 = -6*a4 - 420*a8 - 7560*a12
# p2 = a4 + 210*a8 + 13860*a12
# p3 = -28*a8 - 6930*a12
# p4 = a8 + 1155*a12
# p5 = -66*a12
# p6 = a12
_Pmat = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0],
])

# da12/da_c for c in {0,1,2} (a0, a4, a8)
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Alternating signs for Fourier: q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute the "I matrix" for moment computation
# I_mat[k, i, j] = I[k + i + j] for k in 0..2, i,j in 0..6
_I3 = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 14:
                _I3[_k, _i, _j] = _I[_idx]

# Precompute sum_j p_j * I[k+i+j] structure for gradient
# S[k, i, j] = I[k + i + j]  (same as _I3 but used differently)


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    coeffs = np.array([a0, a4, a8, a12])
    p = _Pmat @ coeffs
    
    # |P|^2 moments: m_k = sum_{i,j} p_i p_j I[k+i+j]
    # Using precomputed _I3
    m = np.zeros(3)
    for k in range(3):
        m[k] = p @ _I3[k] @ p
    
    # Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * (q @ _I3[k] @ q)
    
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
    # dp/da_c = _Pmat[:, c] + _Pmat[:, 3] * da12[c]  for c in 0,1,2
    # where column 3 is the a12 column
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _Pmat[:, c] + _Pmat[:, 3] * _DA12[c]
        
        # dm_k/da_c = 2 * dp @ (_I3[k] @ p)
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * (dp @ (_I3[k] @ p))
        
        dq = dp * _SIGNS
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 2.0 * 2.0 * np.pi * (dq @ (_I3[k] @ q))
        
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


def _coord_descent_polish(x0, max_iter=200, tol=1e-15):
    """Simple coordinate descent with line search for final polish."""
    x = np.array(x0, dtype=float)
    fx, _ = _bound_and_grad(x)
    if not np.isfinite(fx):
        return x, fx
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        for c in range(3):
            # Try small steps in each direction
            for scale in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for sign in [1.0, -1.0]:
                    x_trial = x.copy()
                    x_trial[c] += sign * scale
                    ft, _ = _bound_and_grad(x_trial)
                    if np.isfinite(ft) and ft < fx - tol:
                        x = x_trial
                        fx = ft
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    return x, fx


def construct():
    # Multiple starting points - more diverse and structured
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.0, 0.02, 0.0001]),
        np.array([-1.3, 0.06, 0.0005]),
    ]
    
    rng = np.random.default_rng(123)
    # More random starts with varied scales
    for i in range(30):
        starts.append(rng.normal(0, 0.3, 3))
    for i in range(10):
        starts.append(rng.normal(0, 1.0, 3))
    
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
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B (tighter)
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            x_cd, f_cd = _coord_descent_polish(best_params, max_iter=500)
            if np.isfinite(f_cd) and f_cd < best_val:
                best_val = f_cd
                best_params = x_cd.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
