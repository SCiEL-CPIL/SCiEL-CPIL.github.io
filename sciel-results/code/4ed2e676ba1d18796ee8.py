"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 175


def _chebyshev_setup(N, L=8.0):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices.
    """
    N = int(N)
    
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]
    x = L * t
    
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w
    
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0
    
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    
    D1 = D / L
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L=8.0):
    """
    Build the quadratic forms for position and momentum variances.
    """
    x, w, D1, D2 = _chebyshev_setup(N, L)
    
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    phis = np.array([phi0, phi4, phi8, phi12])
    
    M_0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
    
    x2 = x**2
    M_pos = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
    
    M_mom = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D2_phi_j = D2 @ phis[j]
            M_mom[i, j] = -np.sum(w * phis[i] * D2_phi_j)
    
    return M_0, M_pos, M_mom


def _bound_and_grad(params, M_0, M_pos, M_mom):
    """
    Compute bound and analytic gradient.
    params = (a0, a4, a8)
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_pos @ a
    Q = a @ M_mom @ a
    
    if N <= 1e-30 or P <= 0 or Q <= 0:
        if not np.isfinite(N) or not np.isfinite(P) or not np.isfinite(Q):
            return 1e18, np.zeros(3)
        return 1e18, np.zeros(3)
    
    bound = np.sqrt(P * Q / (N * N))
    
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a
    
    dN_dparams = 2.0 * (da_dparams @ M0a)
    dP_dparams = 2.0 * (da_dparams @ Mposa)
    dQ_dparams = 2.0 * (da_dparams @ Mmoma)
    
    dPQN2 = (dP_dparams * Q + P * dQ_dparams) / (N * N) - 2.0 * P * Q * dN_dparams / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(3)
    
    return bound, grad


def construct():
    N = 256
    
    try:
        M_0, M_pos, M_mom = _build_functional(N)
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val, grad
    
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.1, 0.035, 0.0003]),
        np.array([-1.15, 0.045, 0.00035]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.18, 0.042, 0.00033]),
        np.array([-1.08, 0.037, 0.00029]),
    ]
    
    rng = np.random.default_rng(12345)
    for i in range(20):
        starts.append(rng.normal(0, 0.3, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_obj_grad, start, method='SLSQP',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_obj_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final SLSQP polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='SLSQP',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
