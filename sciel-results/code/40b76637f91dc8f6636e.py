"""Structural rewrite: normalized direction + trust-region + global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200
_SEED_VAL = 1.0  # seed program score reference


def _precompute():
    """Precompute moment integrals and coefficient matrices."""
    sp = np.sqrt(np.pi)
    
    # I[n] = ∫ x^(2n) e^{-x^2} dx = (2n-1)!! / 2^n * sqrt(pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / 2.0
    
    # After substituting a12 = -(a0 + 3*a4 + 105*a8)/945 to enforce g(0)=0:
    # p_k = sum of coefficients * Hermite expansion
    # The 7 coefficients p_0..p_6 are linear in (a0, a4, a8)
    P_mat = np.array([
        [0.0, 0.0, 0.0],           # p0 = 0 (g(0)=0)
        [8.0, 18.0, 420.0],         # p1
        [-44.0/3.0, -43.0, -1330.0], # p2
        [22.0/3.0, 22.0, 742.0],     # p3
        [-11.0/9.0, -11.0, -112.0/9.0], # p4
        [22.0/315.0, 22.0/105.0, 22.0/3.0], # p5
        [-1.0/945.0, -1.0/315.0, -1.0/9.0], # p6
    ])
    
    sign_vec = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    Q_mat = P_mat * sign_vec[:, np.newaxis]
    
    # Moment matrices M_k[i,j] = I[k+i+j] for k=0,2 (variance terms)
    M0 = np.zeros((7, 7))
    M2 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            M0[i, j] = I[i + j]
            M2[i, j] = I[2 + i + j]
    
    return P_mat, Q_mat, M0, M2


_P_mat, _Q_mat, _M0, _M2 = _precompute()


def _evaluate(a0, a4, a8):
    """Compute the uncertainty bound for given coefficients."""
    params = np.array([a0, a4, a8])
    p = _P_mat @ params
    q = _Q_mat @ params
    
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _objective(params):
    """Objective: uncertainty bound."""
    return _evaluate(params[0], params[1], params[2])


def _objective_grad(params):
    """Objective with analytic gradient."""
    a0, a4, a8 = params[0], params[1], params[2]
    params3 = np.array([a0, a4, a8])
    
    p = _P_mat @ params3
    q = _Q_mat @ params3
    
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 0 or n0 <= 0 or m2 <= 0 or n2 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if bound <= 0 or not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    grad = np.zeros(3)
    Mp0 = _M0 @ p
    Mp2 = _M2 @ p
    Mq0 = _M0 @ q
    Mq2 = _M2 @ q
    
    for c in range(3):
        dp = _P_mat[:, c]
        dq = _Q_mat[:, c]
        
        dm0 = 2.0 * (Mp0 @ dp)
        dm2 = 2.0 * (Mp2 @ dp)
        dn0 = 4.0 * np.pi * (Mq0 @ dq)
        dn2 = 4.0 * np.pi * (Mq2 @ dq)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _local_optimize(start, deadline_margin=10):
    """Run local optimization from a starting point with deadline guard."""
    if time.time() > _DEADLINE - deadline_margin:
        return None, None
    
    best_val = _objective(start)
    best_params = start.copy()
    
    # Phase 1: L-BFGS-B with gradient
    try:
        res = minimize(_objective_grad, start, method='L-BFGS-B',
                      jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Trust-region reflective
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_objective_grad, best_params, method='trust-ncg',
                          jac=True, options={'maxiter': 2000, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with Nelder-Mead
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_val, best_params


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 200
    
    best_val = np.inf
    best_params = None
    
    # Seed starting point
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [seed]
    
    # Systematic grid around seed
    for da0 in np.linspace(-0.5, 0.5, 5):
        for da4 in np.linspace(-0.02, 0.02, 5):
            for da8 in np.linspace(-0.0002, 0.0002, 5):
                starts.append(seed + np.array([da0, da4, da8]))
    
    # Random perturbations at various scales
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            starts.append(seed + rng.normal(0, scale, 3))
    
    # Far-field explorations
    for _ in range(20):
        starts.append(rng.normal(0, 3.0, 3))
    
    # Evaluate seed first
    seed_val = _objective(seed)
    best_val = seed_val
    best_params = seed.copy()
    
    # Global search with differential evolution (bounded)
    if time.time() < _DEADLINE - 60:
        try:
            bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            de_res = differential_evolution(
                _objective, bounds, seed=42, maxiter=100, 
                popsize=15, tol=1e-12, polish=False,
                init='sobol', mutation=(0.5, 1.5), recombination=0.9
            )
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Local optimization from best global result
    if best_params is not None and time.time() < _DEADLINE - 30:
        val, params = _local_optimize(best_params, deadline_margin=20)
        if val is not None and val < best_val:
            best_val = val
            best_params = params
    
    # Local optimization from multiple good starting points
    candidate_starts = [s for s in starts if _objective(s) < best_val * 1.5]
    candidate_starts.sort(key=lambda s: _objective(s))
    candidate_starts = candidate_starts[:20]  # Top 20
    
    for start in candidate_starts:
        if time.time() > _DEADLINE - 15:
            break
        val, params = _local_optimize(start, deadline_margin=10)
        if val is not None and val < best_val:
            best_val = val
            best_params = params
    
    # Final refinement if we have a good candidate
    if best_params is not None and time.time() < _DEADLINE - 5:
        # Try a few more local refinements from slightly perturbed best
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            perturbed = best_params + rng.normal(0, 0.01, 3)
            val, params = _local_optimize(perturbed, deadline_margin=5)
            if val is not None and val < best_val:
                best_val = val
                best_params = params
    
    if best_params is None:
        return seed
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
