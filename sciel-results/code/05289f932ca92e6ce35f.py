import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280

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

def _compute_bound(params):
    """Compute the upper bound on C4 given (a0, a4, a8)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    bound = np.sqrt(vf * vF)
    return bound

def _compute_bound_and_grad(params):
    """Compute the upper bound and its gradient with respect to (a0, a4, a8)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
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

def _ratio_objective(uv):
    """
    Objective in ratio variables: u = a4/a0, v = a8/a0.
    Since the bound is scale-invariant (depends only on ratios),
    we set a0 = 1 and compute with a4 = u, a8 = v.
    """
    a0, u, v = 1.0, uv[0], uv[1]
    params = np.array([a0, u, v])
    return _compute_bound(params)

def _ratio_objective_grad(uv):
    """Gradient of the ratio objective (chain rule from 3D to 2D)."""
    a0, u, v = 1.0, uv[0], uv[1]
    params = np.array([a0, u, v])
    val, grad = _compute_bound_and_grad(params)
    # d/d(u) of bound = grad[1] (since a4 = u, a0 fixed)
    # d/d(v) of bound = grad[2] (since a8 = v, a0 fixed)
    # Note: a12 depends on a0, a4, a8, but we've already accounted for that
    # in the 3D gradient computation.
    ratio_grad = np.array([grad[1], grad[2]])
    return val, ratio_grad

def construct():
    best_val = np.inf
    best_params = None
    rng = np.random.default_rng(42)
    
    def _update_best(val, params):
        nonlocal best_val, best_params
        if val < best_val and np.isfinite(val):
            best_val = val
            best_params = np.array(params).copy()
    
    def _try_local_optimize(start, maxiter=3000, ftol=1e-15, gtol=1e-13):
        if time.time() > _DEADLINE - 10:
            return
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 1: Global search using differential evolution in ratio space
    if time.time() < _DEADLINE - 40:
        # Define bounds for ratio variables
        # u = a4/a0, v = a8/a0
        # From the seed, a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
        # So u ~ -0.035, v ~ -0.00027
        # Use wide bounds to ensure global coverage
        bounds_ratio = [(-0.5, 0.5), (-0.01, 0.01)]
        
        try:
            de_result = differential_evolution(
                _ratio_objective,
                bounds=bounds_ratio,
                seed=42,
                maxiter=200,
                tol=1e-12,
                popsize=20,
                polish=True,
                init='sobol',
                updating='deferred'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                u_opt, v_opt = de_result.x
                params_opt = np.array([1.0, u_opt, v_opt])
                _update_best(de_result.fun, params_opt)
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization in original 3D space
    diverse_starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-0.3, 0.01, 0.0001]),
    ]
    
    for start in diverse_starts:
        if time.time() > _DEADLINE - 20:
            break
        _try_local_optimize(start)
    
    # Phase 3: If DE found a good point, use it as additional start
    if best_params is not None and time.time() < _DEADLINE - 15:
        # Try from the best found so far
        _try_local_optimize(best_params, maxiter=5000, ftol=1e-16, gtol=1e-14)
    
    # Phase 4: Random perturbations around best
    if best_params is not None and time.time() < _DEADLINE - 10:
        for scale in [0.1, 0.05, 0.01, 0.005]:
            for _ in range(5):
                if time.time() > _DEADLINE - 5:
                    break
                perturbation = scale * rng.normal(0, 1, 3)
                start = best_params + perturbation
                _try_local_optimize(start, maxiter=1000)
    
    # Phase 5: High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 3:
        _try_local_optimize(best_params, maxiter=10000, ftol=1e-17, gtol=1e-15)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
