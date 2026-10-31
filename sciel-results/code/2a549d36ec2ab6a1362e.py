"""Analytic uncertainty bound exploiting homogeneity to reduce to 2D ratio search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Precompute index matrices for efficiency
_IDX = [k + np.add.outer(np.arange(7), np.arange(7)) for k in range(3)]
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _get_p(a0, a4, a8):
    """Get the coefficient vector p from the three free parameters."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    return p


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty bound and its gradient with respect to (a0, a4, a8).
    Uses the homogeneity to work in ratio space when needed.
    """
    p = _get_p(a0, a4, a8)
    I = _I
    
    pp = np.outer(p, p)
    m0 = np.sum(pp * I[_IDX[0]])
    m2 = np.sum(pp * I[_IDX[1]])
    
    q = p * _SIGNS
    qq = np.outer(q, q)
    n0 = 2.0 * np.pi * np.sum(qq * I[_IDX[0]])
    n2 = 2.0 * np.pi * np.sum(qq * I[_IDX[1]])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute S_p and S_q for gradient
    S_p = np.zeros((7,))
    S_q = np.zeros((7,))
    for i in range(7):
        s_p = 0.0
        s_q = 0.0
        for j in range(7):
            idx = i + j
            if idx < 14:
                s_p += p[j] * I[idx]
                s_q += q[j] * I[idx]
        S_p[i] = s_p
        S_q[i] = s_q
    
    # dp/da matrix: rows are derivatives w.r.t. a0, a4, a8; columns are p components
    dpda = np.zeros((3, 7))
    # a12 = -(a0 + 3*a4 + 105*a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    # p[0] = 0
    # p[1] = -6*a4 - 420*a8 - 7560*a12
    dpda[0, 1] = -7560.0 * (-1.0/945.0)
    dpda[1, 1] = -6.0 - 7560.0 * (-3.0/945.0)
    dpda[2, 1] = -420.0 - 7560.0 * (-105.0/945.0)
    
    # p[2] = a4 + 210*a8 + 13860*a12
    dpda[0, 2] = 13860.0 * (-1.0/945.0)
    dpda[1, 2] = 1.0 + 13860.0 * (-3.0/945.0)
    dpda[2, 2] = 210.0 + 13860.0 * (-105.0/945.0)
    
    # p[3] = -28*a8 - 6930*a12
    dpda[0, 3] = -6930.0 * (-1.0/945.0)
    dpda[1, 3] = -6930.0 * (-3.0/945.0)
    dpda[2, 3] = -28.0 - 6930.0 * (-105.0/945.0)
    
    # p[4] = a8 + 1155*a12
    dpda[0, 4] = 1155.0 * (-1.0/945.0)
    dpda[1, 4] = 1155.0 * (-3.0/945.0)
    dpda[2, 4] = 1.0 + 1155.0 * (-105.0/945.0)
    
    # p[5] = -66*a12
    dpda[0, 5] = -66.0 * (-1.0/945.0)
    dpda[1, 5] = -66.0 * (-3.0/945.0)
    dpda[2, 5] = -66.0 * (-105.0/945.0)
    
    # p[6] = a12
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm0 = 2.0 * (dp @ S_p)
        dm2 = 2.0 * (dp @ S_p)  # Wait, this is wrong. Let me recalculate.
        
        # Actually, m0 = sum_{i,j} p[i]*p[j]*I[i+j]
        # dm0/da_c = sum_{i,j} (dp[i]/da_c * p[j] + p[i] * dp[j]/da_c) * I[i+j]
        #           = 2 * sum_i dp[i] * sum_j p[j]*I[i+j] = 2 * sum_i dp[i] * S_p[i]
        
        # m2 = sum_{i,j} p[i]*p[j]*I[i+j+1]
        # dm2/da_c = 2 * sum_i dp[i] * sum_j p[j]*I[i+j+1]
        
        # Let me recompute S_p for m2 as well
        S_p_m2 = np.zeros(7)
        S_q_n0 = np.zeros(7)
        S_q_n2 = np.zeros(7)
        for i in range(7):
            sp_m2 = 0.0
            sq_n0 = 0.0
            sq_n2 = 0.0
            for j in range(7):
                if i+j < 14:
                    sp_m2 += p[j] * I[i+j+1]
                    sq_n0 += q[j] * I[i+j]
                if i+j+1 < 14:
                    sq_n2 += q[j] * I[i+j+2]
            S_p_m2[i] = sp_m2
            S_q_n0[i] = sq_n0
            S_q_n2[i] = sq_n2
        
        dm0_c = 2.0 * (dp @ S_p)
        dm2_c = 2.0 * (dp @ S_p_m2)
        
        dq = dp * _SIGNS
        dn0_c = 4.0 * np.pi * (dq @ S_q_n0)
        dn2_c = 4.0 * np.pi * (dq @ S_q_n2)
        
        dvm2 = (dm2_c * m0 - m2 * dm0_c) / (m0 * m0)
        dvn2 = (dn2_c * n0 - n2 * dn0_c) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params[0], params[1], params[2])
    return val


def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params[0], params[1], params[2])
    return val, grad


def _bound_from_ratios(r, s):
    """Compute bound using ratios r = a4/a0, s = a8/a0 with a0 = 1."""
    return _objective(np.array([1.0, r, s]))


def construct():
    """
    Exploit homogeneity: the bound depends only on ratios (a4/a0, a8/a0).
    Strategy:
    1. Coarse 2D grid over (r, s) to find promising regions
    2. Local L-BFGS-B refinement from best grid points
    3. Final high-precision polish
    """
    best_val = np.inf
    best_params = None
    
    # Seed point
    seed = np.array([-1.13, 0.04, 0.000308642])
    r_seed = seed[1] / seed[0] if seed[0] != 0 else 0.0
    s_seed = seed[2] / seed[0] if seed[0] != 0 else 0.0
    
    # Phase 1: Coarse 2D grid search over ratios
    r_range = np.linspace(-0.2, 0.2, 31)
    s_range = np.linspace(-0.002, 0.002, 31)
    
    grid_best_val = np.inf
    grid_best_rs = None
    
    for r in r_range:
        if time.time() > _DEADLINE - 120:
            break
        for s in s_range:
            val = _bound_from_ratios(r, s)
            if val < grid_best_val:
                grid_best_val = val
                grid_best_rs = (r, s)
    
    # Phase 2: Local optimization from multiple starting points
    start_points = []
    
    # Add seed
    start_points.append(seed.copy())
    
    # Add grid best
    if grid_best_rs is not None:
        r_best, s_best = grid_best_rs
        start_points.append(np.array([1.0, r_best, s_best]))
        start_points.append(np.array([-1.0, -r_best, -s_best]))
    
    # Add neighborhood around seed in ratio space
    rng = np.random.default_rng(42)
    for _ in range(20):
        dr = rng.normal(0, 0.05)
        ds = rng.normal(0, 0.001)
        r_try = r_seed + dr
        s_try = s_seed + ds
        start_points.append(np.array([1.0, r_try, s_try]))
        start_points.append(np.array([-1.0, -r_try, -s_try]))
    
    # Add more diverse starts
    for r in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
        for s in [-0.0015, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0015]:
            start_points.append(np.array([1.0, r, s]))
            start_points.append(np.array([-1.0, -r, -s]))
    
    # Run L-BFGS-B from each start
    for sp in start_points:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(_objective_grad, sp, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return seed.copy()
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
