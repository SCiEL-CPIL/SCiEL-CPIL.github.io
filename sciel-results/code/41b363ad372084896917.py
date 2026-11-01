import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention
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
            from scipy.special import gammaln
            log_num = gammaln(2*n + 1)
            log_den = n * math.log(4) + gammaln(n + 1)
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
    """
    Computes the bound and gradient for 2D parameters [a4, a8] with a0=1.
    """
    a4, a8 = params[0], params[1]
    a0 = 1.0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    
    m0 = p @ _M_SPATIAL[0] @ p
    m2 = p @ _M_SPATIAL[1] @ p
    
    q = p * _SIGNS
    n0 = q @ _M_FREQ[0] @ q
    n2 = q @ _M_FREQ[1] @ q
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # da12/da4 = -12/665280, da12/da8 = -1680/665280
    da12_da4 = -12.0 / 665280.0
    da12_da8 = -1680.0 / 665280.0
    
    dp_da4 = _H_COEFFS[1] + da12_da4 * _H_COEFFS[3]
    dp_da8 = _H_COEFFS[2] + da12_da8 * _H_COEFFS[3]
    
    grad = np.zeros(2)
    
    # Gradient w.r.t a4
    dm0_4 = 2.0 * (dp_da4 @ _M_SPATIAL[0] @ p)
    dm2_4 = 2.0 * (dp_da4 @ _M_SPATIAL[1] @ p)
    dq_da4 = dp_da4 * _SIGNS
    dn0_4 = 2.0 * (dq_da4 @ _M_FREQ[0] @ q)
    dn2_4 = 2.0 * (dq_da4 @ _M_FREQ[1] @ q)
    
    dvm2_4 = (dm2_4*m0 - m2*dm0_4) / (m0*m0)
    dvn2_4 = (dn2_4*n0 - n2*dn0_4) / (n0*n0)
    grad[0] = (vF * dvm2_4 + vf * dvn2_4) / (2.0 * bound)
    
    # Gradient w.r.t a8
    dm0_8 = 2.0 * (dp_da8 @ _M_SPATIAL[0] @ p)
    dm2_8 = 2.0 * (dp_da8 @ _M_SPATIAL[1] @ p)
    dq_da8 = dp_da8 * _SIGNS
    dn0_8 = 2.0 * (dq_da8 @ _M_FREQ[0] @ q)
    dn2_8 = 2.0 * (dq_da8 @ _M_FREQ[1] @ q)
    
    dvm2_8 = (dm2_8*m0 - m2*dm0_8) / (m0*m0)
    dvn2_8 = (dn2_8*n0 - n2*dn0_8) / (n0*n0)
    grad[1] = (vF * dvm2_8 + vf * dvn2_8) / (2.0 * bound)
    
    return bound, grad

def _objective(x):
    val, _ = _compute_bound_and_grad(x)
    return val

def _objective_grad(x):
    val, grad = _compute_bound_and_grad(x)
    return val, grad

def construct():
    best_val = np.inf
    best_params_2d = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-13):
        nonlocal best_val, best_params_2d
        if time.time() > _DEADLINE - 5:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective, start, jac=_objective_grad, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Powell':
                res = minimize(_objective, start, method='Powell',
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-14})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_2d = res.x
        except Exception:
            pass

    # Known good starting point from seed (normalized a0=1)
    # Seed: a0=-1.13, a4=0.04, a8=0.000308642
    # Normalized: a4_norm = 0.04 / -1.13, a8_norm = 0.000308642 / -1.13
    seed_start = np.array([0.04 / -1.13, 0.000308642 / -1.13])
    
    # Generate a set of starting points
    starts = [
        seed_start,
        np.array([0.0, 0.0]),
        np.array([-0.05, -0.0005]),
        np.array([0.05, 0.0005]),
        np.array([-0.1, -0.001]),
        np.array([0.1, 0.001]),
        np.array([-0.2, -0.002]),
        np.array([0.2, 0.002]),
        np.array([-0.5, -0.005]),
        np.array([0.5, 0.005]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        a4 = rng.uniform(-1.0, 1.0)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a4, a8]))
    
    # Phase 1: Broad search with L-BFGS-B
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, 'L-BFGS-B', 5000)
        
    # Phase 2: Refine the best solution found
    if best_params_2d is not None and time.time() < _DEADLINE - 10:
        # High precision polish
        _try_optimize(best_params_2d, 'L-BFGS-B', 10000, ftol=1e-16, gtol=1e-14)
        
        # Try Powell on the best solution for robustness
        _try_optimize(best_params_2d, 'Powell', 5000)
        
        # Perturbation search around the best
        for scale in [0.1, 0.01, 0.001]:
            if time.time() > _DEADLINE - 10:
                break
            for _ in range(10):
                if time.time() > _DEADLINE - 10:
                    break
                perturbation = scale * np.array([1.0, 0.01]) * rng.normal(0, 1, 2)
                start = best_params_2d + perturbation
                _try_optimize(start, 'L-BFGS-B', 2000)

    # If we found a solution, return the 3D coefficients [a0, a4, a8]
    if best_params_2d is not None:
        a4, a8 = best_params_2d
        a0 = 1.0
        return np.array([a0, a4, a8])
    
    # Fallback
    return np.array([1.0, seed_start[0], seed_start[1]])
