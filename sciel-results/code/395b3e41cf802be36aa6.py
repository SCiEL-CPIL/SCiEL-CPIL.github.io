"""
Analytic uncertainty bound with differential-evolution + L-BFGS-B polishing.

The structural change here is to use a population-based global optimizer
(differential evolution) as the primary search, followed by local polishing.
This is a different optimization/search method from the pure gradient-based
multi-start approach in the seed.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # safety margin before 210s kill


def _precompute_constants():
    """Precompute all constant matrices and vectors needed for fast evaluation."""
    sp = np.sqrt(np.pi)
    
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / 2.0
    
    # p_k as linear functions of (a0, a4, a8) after substituting a12
    # a12 = -(a0 + 3*a4 + 105*a8) / 945
    P_mat = np.zeros((7, 3))
    P_mat[0, :] = [0.0, 0.0, 0.0]
    P_mat[1, :] = [8.0, 18.0, 420.0]
    P_mat[2, :] = [-44.0/3.0, -43.0, -1330.0]
    P_mat[3, :] = [22.0/3.0, 22.0, 742.0]
    P_mat[4, :] = [-11.0/9.0, -11.0, -112.0/9.0]
    P_mat[5, :] = [22.0/315.0, 22.0/105.0, 22.0/3.0]
    P_mat[6, :] = [-1.0/945.0, -1.0/315.0, -1.0/9.0]
    
    # q_k = p_k * (-1)^k
    sign_vec = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    Q_mat = P_mat * sign_vec[:, np.newaxis]
    
    # Build moment matrices:
    # m_k = p^T M_k p where M_k[i,j] = I[k+i+j]
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    M[k, i, j] = I[idx]
    
    return P_mat, Q_mat, M


# Precompute at module level
_P_mat, _Q_mat, _M_mat = _precompute_constants()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient using precomputed constants.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    params3 = np.array([a0, a4, a8])
    
    # p = P_mat @ params3
    p = _P_mat @ params3
    
    # q = Q_mat @ params3
    q = _Q_mat @ params3
    
    # m_k = p^T M_k p
    m0 = p @ _M_mat[0] @ p
    m2 = p @ _M_mat[1] @ p
    
    # n_k = 2*pi * q^T M_k q
    n0 = 2.0 * np.pi * (q @ _M_mat[0] @ q)
    n2 = 2.0 * np.pi * (q @ _M_mat[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient:
    grad = np.zeros(3)
    Mp0 = _M_mat[0] @ p
    Mp1 = _M_mat[1] @ p
    Mq0 = _M_mat[0] @ q
    Mq1 = _M_mat[1] @ q
    
    for c in range(3):
        dp = _P_mat[:, c]
        dq = _Q_mat[:, c]
        
        dm0 = 2.0 * (Mp0 @ dp)
        dm2 = 2.0 * (Mp1 @ dp)
        
        dn0 = 2.0 * 2.0 * np.pi * (Mq0 @ dq)
        dn2 = 2.0 * 2.0 * np.pi * (Mq1 @ dq)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _polish(params):
    """Polish a candidate with L-BFGS-B then Nelder-Mead."""
    best = params.copy()
    best_val = _objective(best)
    
    # L-BFGS-B polish
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    
    return best, best_val


def construct():
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    best_params = seed_params.copy()
    best_val = _objective(best_params)
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < _DEADLINE - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=False,
                updating='immediate',
                init='sobol'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B from diverse points
    starts = [
        best_params.copy(),
        seed_params.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-1.0, -0.05, 0.0005]),
        np.array([1.0, 0.05, -0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.5, -0.02, 0.0003]),
        np.array([2.0, -0.08, -0.001]),
        np.array([-1.5, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(7)
    for i in range(20):
        starts.append(rng.normal(0, 1.0, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            p, v = _polish(start)
            if v < best_val and np.isfinite(v):
                best_val = v
                best_params = p.copy()
        except Exception:
            pass
    
    # Phase 3: Final intensive polish from best
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            p, v = _polish(best_params)
            if v < best_val and np.isfinite(v):
                best_val = v
                best_params = p.copy()
        except Exception:
            pass
    
    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed_params
    
    return np.array(best_params)
