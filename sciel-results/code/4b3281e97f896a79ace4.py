"""Combined uncertainty bound optimization: polynomial parameterization + multi-start + DE."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..30."""
    I = np.zeros(31)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 31):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 is chosen so P(0)=0.
    """
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    # P(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0
    # => a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # P(x) = sum_{k=0}^{6} p_k x^(2k)
    p = np.zeros(7)
    # p[0] = a0 + 3*a4 + 105*a8 + 945*a12 = 0 (by construction)
    p[0] = 0.0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(params):
    """
    Compute the uncertainty product sqrt(vf * vF).
    vf = m2/m0 where m_k = integral x^k |P|^2 e^(-x^2) dx
    vF = n2/n0 where n_k = 2*pi * integral x^k |Q|^2 e^(-x^2) dx, Q_k = (-1)^k p_k
    """
    a0, a4, a8 = params[0], params[1], params[2]
    p = _coeffs_from_params(a0, a4, a8)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute m_k and n_k for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        if p[i] == 0.0 and q[i] == 0.0:
            continue
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 31:
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


def _objective(params):
    return _compute_bound(params)


def _objective_grad(params):
    bound = _compute_bound(params)
    # Use central finite differences for gradient
    h = 1e-7
    grad = np.zeros(3)
    for c in range(3):
        params_p = params.copy()
        params_m = params.copy()
        params_p[c] += h
        params_m[c] -= h
        fp = _compute_bound(params_p)
        fm = _compute_bound(params_m)
        grad[c] = (fp - fm) / (2.0 * h)
    return bound, grad


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad random sampling to find good starting points
    starts = []
    
    # Around the seed
    for _ in range(20):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.15, 3)
        starts.append(s)
    
    # Wider exploration
    for _ in range(50):
        s = np.array([rng.uniform(-3.5, 2.5), rng.uniform(-0.6, 0.6), rng.uniform(-0.015, 0.015)])
        starts.append(s)
    
    # Structured grid
    for a0 in np.linspace(-3.0, 1.5, 12):
        for a4 in np.linspace(-0.4, 0.4, 8):
            a8 = 0.0008 * rng.normal()
            starts.append(np.array([a0, a4, a8]))
    
    # Additional heuristic points
    starts.append(np.array([-1.0, 0.0, 0.0]))
    starts.append(np.array([-1.5, 0.0, 0.0]))
    starts.append(np.array([-2.0, 0.0, 0.0]))
    starts.append(np.array([-0.5, 0.0, 0.0]))
    
    # Evaluate all starts and score them
    scored_starts = []
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val = _compute_bound(s)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = s.copy()
            scored_starts.append((val, s))
        except Exception:
            pass
    
    scored_starts.sort(key=lambda x: x[0])
    
    # Phase 2: L-BFGS-B from top starting points
    top_starts = [s for _, s in scored_starts[:15]]
    top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global search
    if time.time() < _DEADLINE - 50:
        try:
            bounds = [(-4.0, 4.0), (-1.5, 1.5), (-0.02, 0.02)]
            de_res = differential_evolution(_objective, bounds, maxiter=200,
                                           popsize=25, tol=1e-12, seed=42,
                                           polish=False, init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if time.time() < _DEADLINE - 15:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200000, 'xatol': 1e-16, 'fatol': 1e-18})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B refinement
    if time.time() < _DEADLINE - 5:
        try:
            final_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                               jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if final_res.fun < best_val and np.isfinite(final_res.fun):
                best_val = final_res.fun
                best_params = final_res.x.copy()
        except Exception:
            pass
    
    if not np.isfinite(best_val):
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params

# EVOLVE-BLOCK-END
