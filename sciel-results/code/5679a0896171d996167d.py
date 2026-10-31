"""Uncertainty inequality optimization: minimize A(f)*A(F) for Hermite-Gaussian test functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195


def _precompute_I():
    """I[n] = ∫ x^(2n) e^(-x²) dx for n=0..12"""
    I = np.zeros(13)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 13):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _get_p(a0, a4, a8):
    """Compute P(x) coefficients with constraint P(0)=0."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = 0.0  # enforced by construction
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(p):
    """
    For g(x) = P(x)e^(-x²), with P = Σ p_k x^(2k):
    A(f)² = ∫x²|g|²dx / ∫|g|²dx = m2/m0
    A(F)² = (1/2π)·∫ξ²|ĝ|²dξ / (1/2π)·∫|ĝ|²dξ
    
    Since ĝ(ξ) ∝ P(ξ)e^(-ξ²) (Fourier of x^(2k)e^(-x²) ∝ x^(2k)e^(-x²) up to sign),
    A(F)² = n2/n0 where n_k uses same p coefficients.
    """
    m = np.zeros(3)
    n = np.zeros(3)
    for i in range(7):
        if p[i] == 0.0:
            continue
        for j in range(7):
            if p[j] == 0.0:
                continue
            pij = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pij * _I[idx]
                    n[k] += pij * _I[idx]
    
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


def _bound(params):
    p = _get_p(params[0], params[1], params[2])
    return _compute_bound(p)


def _bound_grad(params):
    f0 = _bound(params)
    h = 1e-7
    grad = np.zeros(3)
    for c in range(3):
        dp = params.copy()
        dp[c] += h
        fp = _bound(dp)
        dm = params.copy()
        dm[c] -= h
        fm = _bound(dm)
        grad[c] = (fp - fm) / (2.0 * h)
    return f0, grad


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Broad exploration with structured + random starts
    rng = np.random.default_rng(123)
    
    candidates = []
    # Seed neighborhood
    for _ in range(15):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.15, 3)
        candidates.append(s)
    # Wider ranges
    for _ in range(30):
        s = np.array([rng.uniform(-4, 3), rng.uniform(-1, 1), rng.uniform(-0.05, 0.05)])
        candidates.append(s)
    # Grid exploration
    for a0 in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for a4 in [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]:
            for a8 in [-0.005, -0.001, 0.0, 0.001, 0.005]:
                candidates.append(np.array([a0, a4, a8]))
    
    # Evaluate all candidates
    scored = []
    for s in candidates:
        if time.time() > _DEADLINE - 40:
            break
        try:
            val = _bound(s)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = s.copy()
            scored.append((val, s))
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Phase 2: L-BFGS-B from top candidates
    top_n = min(12, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 25:
            break
        start = scored[i][1].copy()
        try:
            res = minimize(_bound_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global search
    if time.time() < _DEADLINE - 35:
        try:
            bounds = [(-4.0, 4.0), (-1.5, 1.5), (-0.1, 0.1)]
            de_res = differential_evolution(_bound, bounds, maxiter=150,
                                           popsize=20, tol=1e-14, seed=42,
                                           polish=False, init='sobol')
            if np.isfinite(de_res.fun) and de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if time.time() < _DEADLINE - 15:
        try:
            nm_res = minimize(_bound, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20})
            if np.isfinite(nm_res.fun) and nm_res.fun < best_val:
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_bound_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if np.isfinite(fin_res.fun) and fin_res.fun < best_val:
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
