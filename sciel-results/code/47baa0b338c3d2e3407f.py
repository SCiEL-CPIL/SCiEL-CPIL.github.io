"""Analytic uncertainty bound with precomputed integrals + log-barrier refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n * n!) * sqrt(pi)
_I_TABLE = np.zeros(28)
for _n in range(28):
    _log_val = 0.5 * math.log(math.pi)
    for _m in range(1, 2*_n+1):
        _log_val += math.log(_m)
    _log_val -= _n * math.log(4.0)
    for _m in range(1, _n+1):
        _log_val -= math.log(_m)
    _I_TABLE[_n] = math.exp(_log_val)

# Precompute coefficient matrix for P(x) = sum_{k=0}^{6} p_k x^{2k}
# p_k in terms of (a0, a4, a8, a12)
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12
# a12 = -(a0 + 3a4 + 105a8)/945

# dp/da = [dp/da0, dp/da4, dp/da8] including a12 chain rule
# da12/da = [-1/945, -3/945, -105/945]
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Coefficient matrix: row k, col c gives dp_k/da_c
_P_COEFF = np.array([
    [1.0, 3.0, 105.0],
    [0.0, -6.0, -420.0],
    [0.0, 1.0, 210.0],
    [0.0, 0.0, -28.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
_P_A12_COEFF = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# Full dp/da matrix including a12 contribution
_DP_DA = _P_COEFF + np.outer(_P_A12_COEFF, _da12)

# Signs for Fourier: q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_DQ_DA = _DP_DA * _SIGNS[:, np.newaxis]

# Precompute I[k+i+j] lookup for k=0,1,2; i,j=0..6
# I_MAT[k, i, j] = I[k+i+j]
_I_MAT = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _I_MAT[_k, _i, _j] = _I_TABLE[_k + _i + _j]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    Uses precomputed integral table and vectorized operations.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Compute p coefficients
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12,
    ])
    
    # q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.sum(np.outer(p, p) * _I_MAT[k])
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0*math.pi * np.sum(np.outer(q, q) * _I_MAT[k])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Gradient: d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * sum_i dp_i/da_c * sum_j p_j I[k+i+j]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP_DA[:, c]
        dq = _DQ_DA[:, c]
        
        # dm_k = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for k in range(3):
            inner = np.sum(p[:, np.newaxis] * _I_MAT[k], axis=1)  # inner[i] = sum_j p_j I[k+i+j]
            dm[k] = 2.0 * np.dot(dp, inner)
        
        dn = np.zeros(3)
        for k in range(3):
            inner_q = np.sum(q[:, np.newaxis] * _I_MAT[k], axis=1)
            dn[k] = 2.0 * 2.0*math.pi * np.dot(dq, inner_q)
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _log_barrier_objective(params, mu=0.01):
    """
    Log-barrier augmented objective: f(x) + mu * (-log(vf) - log(vF))
    This guides the optimizer away from regions where variances approach zero.
    """
    val, grad = _bound_and_grad(params)
    if not np.isfinite(val) or val >= 1e17:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12,
    ])
    q = p * _SIGNS
    
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.sum(np.outer(p, p) * _I_MAT[k])
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0*math.pi * np.sum(np.outer(q, q) * _I_MAT[k])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or m2 <= 0 or n2 <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    barrier_val = mu * (-math.log(vf) - math.log(vF))
    total_val = val + barrier_val
    
    # Barrier gradient: d(barrier)/da_c = -mu * (d(vf)/vf + d(vF)/vF)
    # d(vf)/da_c = (dm2*m0 - m2*dm0)/m0^2, so d(log(vf))/da_c = d(vf)/(vf*m0) ... 
    # Actually d(log(vf))/da_c = (1/vf) * d(vf)/da_c = (dm2*m0 - m2*dm0)/(vf*m0^2) = (dm2*m0 - m2*dm0)/(m2*m0)
    
    bgrad = np.zeros(3)
    for c in range(3):
        dp = _DP_DA[:, c]
        dq = _DQ_DA[:, c]
        
        dm = np.zeros(3)
        for k in range(3):
            inner = np.sum(p[:, np.newaxis] * _I_MAT[k], axis=1)
            dm[k] = 2.0 * np.dot(dp, inner)
        
        dn = np.zeros(3)
        for k in range(3):
            inner_q = np.sum(q[:, np.newaxis] * _I_MAT[k], axis=1)
            dn[k] = 2.0 * 2.0*math.pi * np.dot(dq, inner_q)
        
        dlog_vf = (dm[1]*m0 - m2*dm[0]) / (m2*m0)
        dlog_vF = (dn[1]*n0 - n2*dn[0]) / (n2*n0)
        
        bgrad[c] = -mu * (dlog_vf + dlog_vF)
    
    total_grad = grad + bgrad
    return total_val, total_grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _log_barrier_grad(params):
    return _log_barrier_objective(params, mu=0.01)


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.3, 0.05, 0.0005]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(25):
        starts.append(rng.normal(0, 0.4, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B multi-start
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Log-barrier refinement from best
    if best_params is not None and time.time() < _DEADLINE - 30:
        # Try decreasing mu for interior-point-like path
        for mu in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() > _DEADLINE - 20:
                break
            try:
                def _lb_obj(x, _mu=mu):
                    return _log_barrier_objective(x, _mu)[0]
                
                def _lb_grad(x, _mu=mu):
                    return _log_barrier_objective(x, _mu)
                
                res = minimize(_lb_grad, best_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13})
                # Evaluate raw objective at the barrier-optimized point
                raw_val, _ = _bound_and_grad(res.x)
                if raw_val < best_val and np.isfinite(raw_val):
                    best_val = raw_val
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
