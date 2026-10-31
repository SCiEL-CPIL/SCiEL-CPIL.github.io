"""Uncertainty inequality optimization with basin-hopping global search and intensive local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 190  # 20s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12."""
    I = np.zeros(13)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 13):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    using probabilists' Hermite polynomials.
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(p):
    """
    Compute the uncertainty product sqrt(vf * vF) given polynomial coeffs p.
    vf = m2/m0 where m_k = integral x^k |P|^2 e^(-x^2) dx
    vF = n2/n0 where n_k = 2*pi * integral x^k |Q|^2 e^(-x^2) dx, Q_k = (-1)^k p_k
    """
    m = np.zeros(3)
    n = np.zeros(3)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            qij = qi * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    return bound


def _bound_from_params(params):
    """Compute bound from (a0, a4, a8)."""
    p = _coeffs_from_params(params[0], params[1], params[2])
    return _compute_bound(p)


def _bound_and_grad(params):
    """Compute bound and analytic gradient via central finite differences."""
    a0, a4, a8 = params
    f0 = _bound_from_params(params)
    h = 1e-7
    grad = np.zeros(3)
    for c in range(3):
        dp = params.copy()
        dp[c] += h
        fp = _bound_from_params(dp)
        dm = params.copy()
        dm[c] -= h
        fm = _bound_from_params(dm)
        grad[c] = (fp - fm) / (2.0 * h)
    return f0, grad


def _objective(params):
    return _bound_from_params(params)


def _objective_grad(params):
    return _bound_and_grad(params)


def _local_optimize(x0, maxiter=5000):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(_objective_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15})
        if res.fun < 1e17 and np.isfinite(res.fun):
            return res.x, res.fun
    except Exception:
        pass
    return x0, _bound_from_params(x0)


def _basin_step(x, scale, rng):
    """Perturb x by a random step of given scale, then locally optimize."""
    x_pert = x + rng.normal(0, scale, 3)
    x_opt, f_opt = _local_optimize(x_pert)
    return x_opt, f_opt


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    t0 = time.time()
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # Around the seed with varying scales
    for scale in [0.05, 0.1, 0.2, 0.5]:
        for _ in range(8):
            s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, scale, 3)
            starts.append(s)
    
    # Broader exploration
    for _ in range(20):
        s = np.array([
            rng.uniform(-3, 2),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.01, 0.01)
        ])
        starts.append(s)
    
    # Structured exploration along each axis
    for a0 in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]:
        for a4 in [-0.15, -0.05, 0.0, 0.05, 0.15]:
            s = np.array([a0, a4, rng.normal(0, 0.001)])
            starts.append(s)
    
    # Evaluate all starts and keep the best ones
    scored = []
    for s in starts:
        if time.time() - t0 > 30:
            break
        try:
            val = _bound_from_params(s)
            if val < 1e17:
                scored.append((val, s))
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    if scored:
        best_val = scored[0][0]
        best_params = scored[0][1].copy()
    
    # Phase 2: Basin hopping with adaptive scale
    # Start with the best points found, use decreasing scales
    top_starts = [s for _, s in scored[:15]]
    if best_params is not None:
        top_starts.append(best_params.copy())
    
    # Basin hopping: perturb and locally optimize
    scales = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    
    current_best = best_params.copy()
    current_val = best_val
    
    for scale in scales:
        if time.time() - t0 > 120:
            break
        
        # Number of trials at this scale, more for larger scales
        n_trials = max(3, int(10 * scale / 0.02))
        n_trials = min(n_trials, 20)
        
        for _ in range(n_trials):
            if time.time() - t0 > 120:
                break
            
            x_pert = current_best + rng.normal(0, scale, 3)
            x_opt, f_opt = _local_optimize(x_pert)
            
            if f_opt < current_val:
                current_val = f_opt
                current_best = x_opt.copy()
                
                if f_opt < best_val:
                    best_val = f_opt
                    best_params = x_opt.copy()
    
    # Phase 3: Intensive local polishing from multiple nearby points
    if time.time() - t0 < 150:
        # Generate a cloud of points around the best
        polish_starts = [best_params.copy()]
        for scale in [0.01, 0.005, 0.001]:
            for _ in range(5):
                polish_starts.append(best_params + rng.normal(0, scale, 3))
        
        for ps in polish_starts:
            if time.time() - t0 > 150:
                break
            x_opt, f_opt = _local_optimize(ps, maxiter=10000)
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
    
    # Phase 4: Final ultra-precise polishing
    if time.time() - t0 < 165:
        for _ in range(3):
            if time.time() - t0 > 165:
                break
            x_opt, f_opt = _local_optimize(best_params, maxiter=20000)
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
    
    # Phase 5: Try Nelder-Mead as a different local optimizer
    if time.time() - t0 < 175:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: One final L-BFGS-B with highest precision
    if time.time() - t0 < 180:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-17})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
