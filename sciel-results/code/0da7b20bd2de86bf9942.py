"""Analytic uncertainty bound with direct polynomial-moment optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _compute_moments(p):
    """
    Given polynomial coefficients p[0..6] for P(x) = sum p_k x^{2k},
    compute moments m_k = integral x^{2k} P(x)^2 e^{-x^2} dx for k=0,1,2
    and Fourier moments n_k = 2*pi * integral x^{2k} Q(x)^2 e^{-x^2} dx
    where Q(x) = P(-ix) = sum p_k (-1)^k x^{2k}.
    """
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    # Precompute I[n] for n=0..12 (max index: 0+6+6=12 for m, same for n)
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # I[n] = I[n-1] * (2*n - 1) / (2*n)
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pji = p[j]
            if pji == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx <= 12:
                    m[k] += pi * pji * I[idx]
    
    # Q coefficients
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qj = q[j]
            if qj == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx <= 12:
                    n[k] += 2.0 * np.pi * qi * qj * I[idx]
    
    return m, n


def _bound_from_params(params):
    """
    Compute the uncertainty bound from (a0, a4, a8).
    Returns (bound, p) where p is the polynomial coefficient vector.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Build polynomial coefficients p[0..6]
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    m, n = _compute_moments(p)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, p
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, p
    
    return np.sqrt(vf * vF), p


def _objective(params):
    val, _ = _bound_from_params(params)
    return val


def _objective_grad(params):
    """
    Compute objective and gradient using numerical differentiation
    with central differences (more robust than analytic for this problem).
    """
    val, _ = _bound_from_params(params)
    
    # Central difference gradient
    h = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        val_plus, _ = _bound_from_params(params_plus)
        val_minus, _ = _bound_from_params(params_minus)
        grad[i] = (val_plus - val_minus) / (2*h)
    
    return val, grad


def construct():
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _objective(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    # Multiple starting points including random ones
    rng = np.random.default_rng(42)
    starts = [
        seed_params.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0003]),
        np.array([-1.3, 0.05, 0.0005]),
        np.array([-0.8, 0.06, 0.0006]),
        np.array([-1.7, 0.07, 0.0007]),
    ]
    
    # Add random starts
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            # L-BFGS-B with numerical gradient
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more Nelder-Mead polish
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
