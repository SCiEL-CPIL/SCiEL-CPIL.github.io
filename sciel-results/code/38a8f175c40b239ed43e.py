from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
import time
import math

# Precompute matrices once at module load (they are constant)
_DEGREES = np.array([0, 4, 8, 12], dtype=np.int64)
_N = 4

def _build_matrices():
    idx = _DEGREES
    M_x = np.zeros((_N, _N))
    M_x2 = np.zeros((_N, _N))
    M_p = np.zeros((_N, _N), dtype=complex)
    M_p2 = np.zeros((_N, _N))

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m + 1:
                M_x[i, j] = np.sqrt(m + 1) / np.sqrt(2)
            elif n == m - 1:
                M_x[i, j] = np.sqrt(m) / np.sqrt(2)

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m + 2:
                M_x2[i, j] = 0.5 * np.sqrt((m + 1) * (m + 2))
            elif n == m:
                M_x2[i, j] = m + 0.5
            elif n == m - 2:
                M_x2[i, j] = 0.5 * np.sqrt(m * (m - 1))

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m - 1:
                M_p[i, j] = 1j * np.sqrt(m) / np.sqrt(2)
            elif n == m + 1:
                M_p[i, j] = -1j * np.sqrt(m + 1) / np.sqrt(2)

    for i in range(_N):
        for j in range(_N):
            if i == j:
                M_p2[i, j] = 2.0 * idx[j] + 1.0
            M_p2[i, j] -= M_x2[i, j]

    return M_x, M_x2, M_p, M_p2

_M_x, _M_x2, _M_p, _M_p2 = _build_matrices()

# Precompute the weight scaling: w_k = c_k * sqrt(2^{d_k} d_k!)
_wscale = np.zeros(_N)
for k in range(_N):
    fact = math.factorial(int(_DEGREES[k]))
    _wscale[k] = np.sqrt((2.0 ** int(_DEGREES[k])) * fact)

# Constraint: g(0) = sum c_n H_n(0) = 0
# H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=-3991680
# So: c0 + 12*c4 + 1680*c8 - 3991680*c12 = 0
# => c12 = (c0 + 12*c4 + 1680*c8) / 3991680
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = -3991680.0

def _c12_from(c4, c8, c0=1.0):
    return (c0 + _H4_0 * c4 + _H8_0 * c8) / (-_H12_0)

def _weights_from_params(c4, c8, c0=1.0):
    c12 = _c12_from(c4, c8, c0)
    c = np.array([c0, c4, c8, c12])
    return c * _wscale

def _uncertainty_product(w):
    """Compute sqrt(var_x * var_p) given orthonormal-basis weights w."""
    norm_sq = float(np.dot(w, w))
    if norm_sq < 1e-15:
        return np.inf
    exp_x = float(np.dot(w, _M_x @ w))
    exp_x2 = float(np.dot(w, _M_x2 @ w))
    exp_p = float(np.dot(w, _M_p @ w).real)
    exp_p2 = float(np.dot(w, _M_p2 @ w))
    var_x = exp_x2 - exp_x * exp_x
    var_p = exp_p2 - exp_p * exp_p
    if var_x < 0: var_x = 0.0
    if var_p < 0: var_p = 0.0
    return math.sqrt(var_x * var_p)

def _objective_and_grad(params):
    """Objective and analytic gradient."""
    c4, c8 = params
    w = _weights_from_params(c4, c8)
    
    exp_x = float(w @ _M_x @ w)
    exp_x2 = float(w @ _M_x2 @ w)
    exp_p = float((w @ _M_p @ w).real)
    exp_p2 = float(w @ _M_p2 @ w)
    
    var_x = max(exp_x2 - exp_x**2, 0.0)
    var_p = max(exp_p2 - exp_p**2, 0.0)
    
    if var_x < 1e-30 or var_p < 1e-30:
        return 1e10, np.zeros(2)
    
    obj = math.sqrt(var_x * var_p)
    
    dvarx_dw = 2.0 * (_M_x2 @ w) - 2.0 * exp_x * (_M_x @ w)
    dvarp_dw = 2.0 * (_M_p2 @ w) - 4.0 * exp_p * np.real(_M_p @ w)
    
    dobj_dw = (dvarx_dw * var_p + var_x * dvarp_dw) / (2.0 * obj)
    
    dw_dc4 = _wscale * np.array([0.0, 1.0, 0.0, _H4_0 / (-_H12_0)])
    dw_dc8 = _wscale * np.array([0.0, 0.0, 1.0, _H8_0 / (-_H12_0)])
    
    grad_c4 = float(dobj_dw @ dw_dc4)
    grad_c8 = float(dobj_dw @ dw_dc8)
    
    return obj, np.array([grad_c4, grad_c8])

def construct():
    start_time = time.time()
    deadline = 195.0  # safety margin before 210s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - elapsed()
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    def try_update(params, score):
        nonlocal best_score, best_params
        if score < best_score and np.isfinite(score):
            best_score = score
            best_params = params.copy()

    # Use differential evolution for global search
    # Bounds are chosen to cover the likely region of the optimum
    bounds = [(-10, 10), (-10, 10)]
    
    try:
        res = differential_evolution(
            lambda p: _objective_and_grad(p)[0],
            bounds=bounds,
            maxiter=500,
            popsize=20,
            tol=1e-12,
            atol=1e-12,
            seed=42,
            polish=True,
            init='sobol'
        )
        if res.fun < best_score and np.isfinite(res.fun):
            try_update(res.x, res.fun)
    except Exception:
        pass
    
    # Local refinement
    if time_left() > 5.0 and np.isfinite(best_score):
        try:
            res = minimize(
                lambda p: _objective_and_grad(p)[0],
                best_params,
                jac=lambda p: _objective_and_grad(p)[1],
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-15}
            )
            if res.fun < best_score:
                try_update(res.x, res.fun)
        except Exception:
            pass
    
    return np.array([1.0, best_params[0], best_params[1]])
