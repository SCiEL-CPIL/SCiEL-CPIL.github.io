import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200 # 10s margin

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Physicists' convention
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-12 or n0 <= 1e-12 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        dq = dp * _SIGNS
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    best_val = np.inf
    best_params = None
    
    # Phase 1: Differential Evolution for global search
    # Bounds: a0, a4, a8
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    def de_callback(xk, convergence):
        nonlocal best_val, best_params
        val, _ = _compute_bound_and_grad(xk)
        if val < best_val and np.isfinite(val):
            best_val = val
            best_params = xk.copy()
        return time.time() > _DEADLINE - 30

    try:
        if time.time() < _DEADLINE - 40:
            de_res = differential_evolution(_objective, bounds, seed=42, maxiter=50, popsize=15, callback=de_callback, tol=1e-12, polish=False)
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
    except Exception:
        pass
    
    # If DE failed or didn't find anything, use the seed point
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
        best_val, _ = _compute_bound_and_grad(best_params)

    # Phase 2: Local Optimization with L-BFGS-B
    def _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-13):
        nonlocal best_val, best_params
        if time.time() > _DEADLINE - 10:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-14})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    _try_optimize(best_params, 'L-BFGS-B', 5000)
    
    # Phase 3: Basin Hopping
    if time.time() < _DEADLINE - 20:
        def _basinhopping_objective(x):
            return _objective(x)
        
        try:
            bh_res = basinhopping(_basinhopping_objective, best_params, niter=50, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True}, stepsize=0.5, seed=42)
            if bh_res.fun < best_val and np.isfinite(bh_res.fun):
                best_val = bh_res.fun
                best_params = bh_res.x.copy()
        except Exception:
            pass

    # Phase 4: Final Polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        _try_optimize(best_params, 'L-BFGS-B', 10000, ftol=1e-16, gtol=1e-14)
        _try_optimize(best_params, 'Nelder-Mead', 5000)

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
