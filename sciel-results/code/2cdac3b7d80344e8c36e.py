"""Uncertainty inequality optimization with analytic gradient and multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution, root

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # should be ~0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


# Precompute dp/da matrix analytically
# a12 = -(a0 + 3*a4 + 105*a8)/945
# da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
_DP_DA = np.zeros((7, 3))
_DP_DA[0] = [0.0, 0.0, 0.0]
# p[1] = -6*a4 - 420*a8 - 7560*a12
_DP_DA[1] = [-7560.0 * (-1.0/945.0), -6.0 - 7560.0 * (-3.0/945.0), -420.0 - 7560.0 * (-105.0/945.0)]
# p[2] = a4 + 210*a8 + 13860*a12
_DP_DA[2] = [13860.0 * (-1.0/945.0), 1.0 + 13860.0 * (-3.0/945.0), 210.0 + 13860.0 * (-105.0/945.0)]
# p[3] = -28*a8 - 6930*a12
_DP_DA[3] = [-6930.0 * (-1.0/945.0), -6930.0 * (-3.0/945.0), -28.0 - 6930.0 * (-105.0/945.0)]
# p[4] = a8 + 1155*a12
_DP_DA[4] = [1155.0 * (-1.0/945.0), 1155.0 * (-3.0/945.0), 1.0 + 1155.0 * (-105.0/945.0)]
# p[5] = -66*a12
_DP_DA[5] = [-66.0 * (-1.0/945.0), -66.0 * (-3.0/945.0), -66.0 * (-105.0/945.0)]
# p[6] = a12
_DP_DA[6] = [-1.0/945.0, -3.0/945.0, -105.0/945.0]

_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(vf * vF) and its analytic gradient
    with respect to (a0, a4, a8).
    """
    p = _coeffs_from_params(a0, a4, a8)
    q = p * _SIGN
    
    # Compute m_k and n_k for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Analytic gradient
    # d(bound)/da_i = (1/(2*bound)) * (d(vf)/da_i * vF + vf * d(vF)/da_i)
    # d(vf)/da_i = (dm2/da_i * m0 - m2 * dm0/da_i) / m0^2
    # d(vF)/da_i = (dn2/da_i * n0 - n2 * dn0/da_i) / n0^2
    # dm_k/da_i = 2 * sum_j p_j * dp_k/da_i * I[k+i+j]  (for each i,j pair, by symmetry)
    # More precisely: dm_k = sum_{i,j} p_i p_j I[k+i+j]
    # dm_k/da_c = sum_{i,j} (dp_i/da_c * p_j + p_i * dp_j/da_c) * I[k+i+j]
    #            = 2 * sum_i dp_i/da_c * sum_j p_j * I[k+i+j]  (by symmetry of I and p_i p_j)
    
    # Precompute S[k][i] = sum_j p_j * I[k+i+j]
    S_m = np.zeros((3, 7))
    S_n = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_m = 0.0
            s_n = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 25:
                    s_m += p[j] * _I[idx]
                    s_n += q[j] * _I[idx]
            S_m[k, i] = s_m
            S_n[k, i] = s_n
    
    grad = np.zeros(3)
    for c in range(3):
        dm0_da = 2.0 * np.dot(_DP_DA[:, c], S_m[0, :])
        dm2_da = 2.0 * np.dot(_DP_DA[:, c], S_m[1, :])
        dn0_da = 2.0 * np.dot(_DP_DA[:, c], S_n[0, :])
        dn2_da = 2.0 * np.dot(_DP_DA[:, c], S_n[1, :])
        
        dvf_da = (dm2_da * m0 - m2 * dm0_da) / (m0 * m0)
        dvF_da = (dn2_da * n0 - n2 * dn0_da) / (n0 * n0)
        
        grad[c] = (dvf_da * vF + vf * dvF_da) / (2.0 * bound)
    
    return bound, grad


def _compute_bound(a0, a4, a8):
    """Compute just the bound value."""
    bound, _ = _compute_bound_and_grad(a0, a4, a8)
    return bound


def _objective(params):
    return _compute_bound(params[0], params[1], params[2])


def _objective_grad(params):
    bound, grad = _compute_bound_and_grad(params[0], params[1], params[2])
    return bound, grad


def _stationary_equations(params):
    """Return gradient of the bound (should be zero at stationary points)."""
    _, grad = _compute_bound_and_grad(params[0], params[1], params[2])
    return grad


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad random sampling
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
    
    # Evaluate all starts
    for s in starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            val = _compute_bound(s[0], s[1], s[2])
            if val < best_val:
                best_val = val
                best_params = s.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B from best starts
    scored_starts = []
    for s in starts:
        try:
            val = _compute_bound(s[0], s[1], s[2])
            scored_starts.append((val, s))
        except Exception:
            pass
    scored_starts.sort(key=lambda x: x[0])
    
    top_starts = [s for _, s in scored_starts[:15]]
    top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Stationary point search via root-finding
    if time.time() < _DEADLINE - 35:
        try:
            # Try from multiple starting points
            for s in top_starts[:8]:
                if time.time() > _DEADLINE - 35:
                    break
                try:
                    sol = root(_stationary_equations, s, method='hybr',
                              options={'xtol': 1e-14, 'maxfev': 5000})
                    if sol.success and np.all(np.abs(sol.x) < 10):
                        val = _compute_bound(sol.x[0], sol.x[1], sol.x[2])
                        if val < best_val and np.isfinite(val):
                            best_val = val
                            best_params = sol.x.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Phase 4: Differential evolution for global search
    if time.time() < _DEADLINE - 30:
        try:
            bounds = [(-4.0, 4.0), (-1.5, 1.5), (-0.02, 0.02)]
            de_res = differential_evolution(_objective, bounds, maxiter=150,
                                           popsize=20, tol=1e-12, seed=42,
                                           polish=False, init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish
    if time.time() < _DEADLINE - 12:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final high-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 50000, 'ftol': 1e-17, 'gtol': 1e-16})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 7: Perturbation and re-optimization
    if time.time() < _DEADLINE - 3:
        try:
            for scale in [1e-3, 1e-4, 1e-5]:
                if time.time() > _DEADLINE - 3:
                    break
                for _ in range(4):
                    perturbed = best_params + rng.normal(0, scale, 3)
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
