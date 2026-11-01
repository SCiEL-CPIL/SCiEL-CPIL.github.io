"""Analytic uncertainty bound with grid search + exact gradient refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(20)
    for n in range(20):
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute the derivative coefficient matrix
# dp_k/da_c for c=0,1,2 (a0, a4, a8)
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
_dp_da = np.zeros((7, 3))

# p0 = a0 + 3a4 + 105a8 + 945a12
_dp_da[0, 0] = 1.0 + 945.0 * _da12[0]
_dp_da[0, 1] = 3.0 + 945.0 * _da12[1]
_dp_da[0, 2] = 105.0 + 945.0 * _da12[2]

# p1 = -6a4 - 420a8 - 7560a12
_dp_da[1, 0] = 0.0 - 7560.0 * _da12[0]
_dp_da[1, 1] = -6.0 - 7560.0 * _da12[1]
_dp_da[1, 2] = -420.0 - 7560.0 * _da12[2]

# p2 = a4 + 210a8 + 13860a12
_dp_da[2, 0] = 0.0 + 13860.0 * _da12[0]
_dp_da[2, 1] = 1.0 + 13860.0 * _da12[1]
_dp_da[2, 2] = 210.0 + 13860.0 * _da12[2]

# p3 = -28a8 - 6930a12
_dp_da[3, 0] = 0.0 - 6930.0 * _da12[0]
_dp_da[3, 1] = 0.0 - 6930.0 * _da12[1]
_dp_da[3, 2] = -28.0 - 6930.0 * _da12[2]

# p4 = a8 + 1155a12
_dp_da[4, 0] = 0.0 + 1155.0 * _da12[0]
_dp_da[4, 1] = 0.0 + 1155.0 * _da12[1]
_dp_da[4, 2] = 1.0 + 1155.0 * _da12[2]

# p5 = -66a12
_dp_da[5, 0] = 0.0 - 66.0 * _da12[0]
_dp_da[5, 1] = 0.0 - 66.0 * _da12[1]
_dp_da[5, 2] = 0.0 - 66.0 * _da12[2]

# p6 = a12
_dp_da[6, 0] = 0.0 + 1.0 * _da12[0]
_dp_da[6, 1] = 0.0 + 1.0 * _da12[1]
_dp_da[6, 2] = 0.0 + 1.0 * _da12[2]

# Precompute (-1)^k for q_k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_pi2 = 2.0 * np.pi

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    I = _I
    
    # Calculate moments m0, m1, m2 for |g|^2
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pp = pi * p[j]
            m0 += pp * I[i+j]
            m1 += pp * I[i+j+1]
            m2 += pp * I[i+j+2]
    
    # Calculate moments n0, n1, n2 for |G|^2
    # q_k = p_k (-1)^k
    q = p * _signs
    
    n0 = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qq = qi * q[j]
            n0 += qq * I[i+j]
            n1 += qq * I[i+j+1]
            n2 += qq * I[i+j+2]
    
    n0 *= _pi2
    n1 *= _pi2
    n2 *= _pi2
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[:, c]
        
        dm0 = 0.0
        dm1 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for j in range(7):
                pj = p[j]
                s0 += pj * I[i+j]
                s1 += pj * I[i+j+1]
                s2 += pj * I[i+j+2]
            dm0 += 2.0 * dpi * s0
            dm1 += 2.0 * dpi * s1
            dm2 += 2.0 * dpi * s2
        
        dq = dp * _signs
        dn0 = 0.0
        dn1 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for j in range(7):
                qj = q[j]
                s0 += qj * I[i+j]
                s1 += qj * I[i+j+1]
                s2 += qj * I[i+j+2]
            dn0 += 2.0 * _pi2 * dqi * s0
            dn1 += 2.0 * _pi2 * dqi * s1
            dn2 += 2.0 * _pi2 * dqi * s2
        
        dvm2 = (dm1*m0 - m1*dm0) / (m0*m0)
        dvn2 = (dn1*n0 - n1*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _vectorized_bound(params_array):
    """
    Vectorized computation of the bound for an array of parameters.
    params_array: shape (N, 3)
    Returns: shape (N,) array of bounds
    """
    N = params_array.shape[0]
    bounds = np.full(N, 1e18)
    
    a0 = params_array[:, 0]
    a4 = params_array[:, 1]
    a8 = params_array[:, 2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # p_k arrays of shape (N,)
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.stack([p0, p1, p2, p3, p4, p5, p6], axis=1)  # (N, 7)
    q = p * _signs  # (N, 7)
    
    I = _I
    
    # m0 = sum_{i,j} p_i p_j I[i+j]
    m0 = np.zeros(N)
    m1 = np.zeros(N)
    m2 = np.zeros(N)
    n0 = np.zeros(N)
    n1 = np.zeros(N)
    n2 = np.zeros(N)
    
    for i in range(7):
        for j in range(7):
            pp = p[:, i] * p[:, j]
            m0 += pp * I[i+j]
            m1 += pp * I[i+j+1]
            m2 += pp * I[i+j+2]
            
            qq = q[:, i] * q[:, j]
            n0 += qq * I[i+j]
            n1 += qq * I[i+j+1]
            n2 += qq * I[i+j+2]
    
    n0 *= _pi2
    n1 *= _pi2
    n2 *= _pi2
    
    valid = (m0 > 0) & (n0 > 0) & np.isfinite(m0) & np.isfinite(n0)
    vf = np.zeros(N)
    vF = np.zeros(N)
    vf[valid] = m1[valid] / m0[valid]
    vF[valid] = n1[valid] / n0[valid]
    
    valid2 = valid & (vf > 0) & (vF > 0)
    bounds[valid2] = np.sqrt(vf[valid2] * vF[valid2])
    
    return bounds

def construct():
    best_val = np.inf
    best_params = None
    
    # Phase 1: Dense grid search over the 2D coefficient space
    # We parameterize by (a4, a8) and compute a0 from the constraint
    # Actually, we have 3 free coefficients (a0, a4, a8) with a12 determined
    # Let's do a grid over a0 and a4, with a8 fixed at a few values, or
    # better: grid over (a4, a8) with a0 as a free parameter
    
    # Let's use a 2D grid in (a0, a4) space with a8 as a third dimension
    # But that's 3D. Instead, let's do a structured search.
    
    # Strategy: Fix a8 at several values, then grid over (a0, a4)
    a8_values = np.array([0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01])
    
    for a8_val in a8_values:
        if time.time() > _DEADLINE - 80:
            break
        
        # Grid over a0 and a4
        a0_vals = np.linspace(-3.0, 1.0, 60)
        a4_vals = np.linspace(-0.2, 0.3, 60)
        
        params_grid = np.array([[a0, a4, a8_val] 
                                for a0 in a0_vals 
                                for a4 in a4_vals])
        
        bounds = _vectorized_bound(params_grid)
        
        min_idx = np.argmin(bounds)
        if bounds[min_idx] < best_val and np.isfinite(bounds[min_idx]):
            best_val = bounds[min_idx]
            best_params = params_grid[min_idx].copy()
    
    # Phase 2: Local refinement from best grid point
    if best_params is not None and time.time() < _DEADLINE - 40:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional starting points for L-BFGS-B
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        scale = 0.2 + 0.5 * i / 20
        starts.append(rng.normal(0, scale, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Polish with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
