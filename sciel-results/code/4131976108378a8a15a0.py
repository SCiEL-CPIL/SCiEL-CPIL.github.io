"""Uncertainty bound minimization with multi-start gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

def _coeffs_to_poly(a0, a4, a8):
    """Convert Hermite coefficients to polynomial coefficients p[0..6] and q[0..6]."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    return p, q

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    params = [a0, a4, a8]
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    p, q = _coeffs_to_poly(a0, a4, a8)
    
    # Compute moments m[k] = sum_{i,j} p_i p_j I[k+i+j]
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute S_p and S_q for gradient
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # dp/da matrix: dpda[c, i] = dp[i]/da[c]
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    dpda = np.zeros((3, 7))
    # p[0] = 0 always
    # p[1] = -6a4 - 420a8 - 7560a12
    dpda[0, 1] = -7560.0 * da12[0]
    dpda[1, 1] = -6.0 - 7560.0 * da12[1]
    dpda[2, 1] = -420.0 - 7560.0 * da12[2]
    # p[2] = a4 + 210a8 + 13860a12
    dpda[0, 2] = 13860.0 * da12[0]
    dpda[1, 2] = 1.0 + 13860.0 * da12[1]
    dpda[2, 2] = 210.0 + 13860.0 * da12[2]
    # p[3] = -28a8 - 6930a12
    dpda[0, 3] = -6930.0 * da12[0]
    dpda[1, 3] = -6930.0 * da12[1]
    dpda[2, 3] = -28.0 - 6930.0 * da12[2]
    # p[4] = a8 + 1155a12
    dpda[0, 4] = 1155.0 * da12[0]
    dpda[1, 4] = 1155.0 * da12[1]
    dpda[2, 4] = 1.0 + 1155.0 * da12[2]
    # p[5] = -66a12
    dpda[0, 5] = -66.0 * da12[0]
    dpda[1, 5] = -66.0 * da12[1]
    dpda[2, 5] = -66.0 * da12[2]
    # p[6] = a12
    dpda[0, 6] = da12[0]
    dpda[1, 6] = da12[1]
    dpda[2, 6] = da12[2]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _local_refine(x0, max_time=5.0):
    """Multi-stage local refinement from a starting point."""
    best_x = x0.copy()
    best_val, _ = _bound_and_grad(x0)
    
    if not np.isfinite(best_val):
        return x0, best_val
    
    # Stage 1: L-BFGS-B with gradient
    t0 = time.time()
    try:
        res = minimize(_objective_grad, x0, method='L-BFGS-B',
                      jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 > max_time:
        return best_x, best_val
    
    # Stage 2: Nelder-Mead polish
    t1 = time.time()
    try:
        res = minimize(_objective, best_x, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t1 > max_time:
        return best_x, best_val
    
    # Stage 3: Another L-BFGS-B from polished point
    t2 = time.time()
    try:
        res = minimize(_objective_grad, best_x, method='L-BFGS-B',
                      jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 50})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_val

def construct():
    seed_coeffs = np.array([-1.13, 0.04, 0.000308642])
    
    best_val = np.inf
    best_x = seed_coeffs.copy()
    
    # Evaluate seed
    seed_val, _ = _bound_and_grad(seed_coeffs)
    if np.isfinite(seed_val):
        best_val = seed_val
    
    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    starts = [seed_coeffs.copy()]
    
    # Grid around seed
    for da0 in np.linspace(-0.5, 0.5, 9):
        for da4 in np.linspace(-0.05, 0.15, 9):
            for da8 in np.linspace(-0.005, 0.01, 7):
                starts.append(seed_coeffs + np.array([da0, da4, da8]))
    
    # Wider grid
    for da0 in np.linspace(-2.0, 2.0, 15):
        for da4 in np.linspace(-0.5, 0.5, 15):
            for da8 in np.linspace(-0.05, 0.05, 9):
                starts.append(seed_coeffs + np.array([da0, da4, da8]))
    
    # Random points in a reasonable range
    for _ in range(200):
        x = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.1, 0.1)
        ])
        starts.append(x)
    
    # Some very different scale points
    for scale in [0.1, 0.5, 2.0, 5.0, 10.0]:
        for _ in range(20):
            x = rng.normal(0, scale, size=3)
            x[2] *= 0.05  # a8 tends to be smaller
            starts.append(x)
    
    # Sort starts by initial objective value (best first) to get good results early
    start_vals = []
    for s in starts:
        v, _ = _bound_and_grad(s)
        start_vals.append(v)
    
    # Sort by value
    order = np.argsort(start_vals)
    starts = [starts[i] for i in order]
    start_vals = [start_vals[i] for i in order]
    
    # Multi-start optimization with time budget
    time_per_start = max(0.5, (180.0 - (time.time() - _START_TIME)) / max(1, len(starts)))
    
    for i, (s, sv) in enumerate(zip(starts, start_vals)):
        if time.time() > _DEADLINE - 15:
            break
        
        if sv >= best_val:
            continue
        
        # Quick L-BFGS-B first
        t0 = time.time()
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 30})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        # If this was a promising start, do more thorough refinement
        if res.fun < best_val + 0.01 and time.time() < _DEADLINE - 30:
            try:
                rx, rv = _local_refine(res.x, max_time=min(8.0, _DEADLINE - time.time() - 20))
                if rv < best_val and np.isfinite(rv):
                    best_val = rv
                    best_x = rx.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            rx, rv = _local_refine(best_x, max_time=min(10.0, _DEADLINE - time.time() - 5))
            if rv < best_val and np.isfinite(rv):
                best_val = rv
                best_x = rx.copy()
        except Exception:
            pass
    
    # Ultra-fine Nelder-Mead
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_x, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
