"""Analytic uncertainty bound with exact gradient + enhanced multi-start + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

_DEADLINE = time.time() + 205

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()
_sp = np.sqrt(np.pi)

# Precompute the index matrices for m and n computation
_ar7 = np.arange(7)
_idx_mat = np.add.outer(_ar7, _ar7)  # 7x7 matrix of i+j
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute the linear map from (a0, a4, a8) to p coefficients
_H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
_H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

_A_mat = np.column_stack([
    _H0_c + _c12[0]*_H12_c,
    _H4_c + _c12[1]*_H12_c,
    _H8_c + _c12[2]*_H12_c
])  # Shape (7, 3)


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    pp = np.outer(p, p)
    m0 = np.sum(pp * I[_idx_mat])
    m2 = np.sum(pp * I[1 + _idx_mat])
    
    # For Fourier: q_k = p_k * (-1)^k
    q = p * signs
    qq = np.outer(q, q)
    n0 = 2.0 * np.pi * np.sum(qq * I[_idx_mat])
    n2 = 2.0 * np.pi * np.sum(qq * I[1 + _idx_mat])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically using the precomputed A_mat
    S_p0 = np.zeros(7)
    S_p1 = np.zeros(7)
    S_q0 = np.zeros(7)
    S_q1 = np.zeros(7)
    for i in range(7):
        idx0 = i + _ar7
        idx1 = 1 + i + _ar7
        S_p0[i] = np.dot(p, I[idx0])
        S_p1[i] = np.dot(p, I[idx1])
        S_q0[i] = np.dot(q, I[idx0])
        S_q1[i] = np.dot(q, I[idx1])
    
    # Gradient: for each parameter c in {0,1,2}, dp = A_mat[:, c]
    grad = np.zeros(3)
    for c in range(3):
        dp = _A_mat[:, c]
        
        dm0 = 2.0 * (dp @ S_p0)
        dm2 = 2.0 * (dp @ S_p1)
        
        dq = dp * signs
        dn0 = 4.0 * np.pi * (dq @ S_q0)
        dn2 = 4.0 * np.pi * (dq @ S_q1)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _try_optimize(start, method='L-BFGS-B', maxiter=5000, **kwargs):
    """Try optimization from a starting point, return (value, params) or None."""
    try:
        if method == 'L-BFGS-B':
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
        elif method == 'Nelder-Mead':
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
        elif method == 'Powell':
            res = minimize(_objective, start, method='Powell',
                          options={'maxiter': maxiter, 'xtol': 1e-16, 'ftol': 1e-18})
        else:
            return None
        if np.isfinite(res.fun):
            return res.fun, res.x.copy()
    except Exception:
        pass
    return None


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Generate diverse starting points
    rng = np.random.default_rng(2024)
    
    starts = [seed.copy()]
    
    # Latin hypercube sampling in a box around the seed
    n_lh = 30
    for i in range(n_lh):
        u0 = rng.uniform(-2.0, 2.0)
        u4 = rng.uniform(-0.1, 0.2)
        u8 = rng.uniform(-0.001, 0.002)
        starts.append(np.array([seed[0] + u0, seed[1] + u4, seed[2] + u8]))
    
    # Random starts with various scales
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for i in range(10):
            pert = rng.normal(0, scale, 3)
            # Scale the perturbation appropriately for each dimension
            pert[1] *= 0.1  # a4 has smaller typical magnitude
            pert[2] *= 0.001  # a8 has much smaller typical magnitude
            starts.append(seed + pert)
    
    # Grid around the seed
    for a0 in [-1.5, -1.3, -1.1, -0.9, -0.7]:
        for a4 in [0.02, 0.03, 0.04, 0.05, 0.06]:
            for a8 in [0.0002, 0.0003, 0.0004, 0.0005]:
                starts.append(np.array([a0, a4, a8]))
    
    # Directional starts
    directions = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1]),
        np.array([1, 1, 1]),
        np.array([-1, 0.5, 0.1]),
        np.array([0.5, -1, 0.2]),
        np.array([0.2, 0.3, -1]),
        np.array([1, -1, 0]),
        np.array([-1, 1, 0]),
    ]
    for d in directions:
        d_norm = d / np.linalg.norm(d)
        for scale in [0.1, 0.5, 1.0, 2.0]:
            starts.append(seed + scale * d_norm)
            starts.append(seed - scale * d_norm)
    
    # Phase 2: Multi-start optimization
    best_val = np.inf
    best_params = None
    local_minima = []
    
    for start in starts:
        if time.time() > _DEADLINE - 80:
            break
        result = _try_optimize(start, method='L-BFGS-B', maxiter=10000)
        if result is not None:
            val, params = result
            if val < best_val:
                best_val = val
                best_params = params
            local_minima.append((val, params))
    
    # Phase 3: Polish the best with Nelder-Mead
    if best_params is not None and time.time() < _DEADLINE - 50:
        result = _try_optimize(best_params, method='Nelder-Mead', maxiter=100000)
        if result is not None:
            val, params = result
            if val < best_val:
                best_val = val
                best_params = params
    
    # Phase 4: Polish with Powell
    if best_params is not None and time.time() < _DEADLINE - 40:
        result = _try_optimize(best_params, method='Powell', maxiter=50000)
        if result is not None:
            val, params = result
            if val < best_val:
                best_val = val
                best_params = params
    
    # Phase 5: High-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Coordinate descent with very tight tolerances
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            bp = best_params.copy()
            improved = True
            while improved and time.time() < _DEADLINE - 10:
                improved = False
                for dim in range(3):
                    def line_obj(t):
                        trial = bp.copy()
                        trial[dim] = t
                        return _objective(trial)
                    
                    # Multi-scale coordinate search
                    for half_width in [1.0, 0.1, 0.01, 0.001, 0.0001]:
                        res1d = minimize_scalar(line_obj, 
                                               bounds=(bp[dim]-half_width, bp[dim]+half_width),
                                               method='bounded', options={'xatol': 1e-15})
                        if np.isfinite(res1d.fun) and res1d.fun < best_val:
                            best_val = res1d.fun
                            bp[dim] = res1d.x
                            improved = True
            best_params = bp
        except Exception:
            pass
    
    # Phase 7: Final ultra-precise L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 200000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 8: Final coordinate descent polish
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            bp = best_params.copy()
            for dim in range(3):
                def line_obj(t):
                    trial = bp.copy()
                    trial[dim] = t
                    return _objective(trial)
                for half_width in [0.01, 0.001, 0.0001, 0.00001]:
                    res1d = minimize_scalar(line_obj,
                                           bounds=(bp[dim]-half_width, bp[dim]+half_width),
                                           method='bounded', options={'xatol': 1e-18})
                    if np.isfinite(res1d.fun) and res1d.fun < best_val:
                        best_val = res1d.fun
                        bp[dim] = res1d.x
            best_params = bp
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
