"""Uncertainty inequality optimization with Remez exchange and analytic gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195.0

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
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p

def _compute_m_n(a0, a4, a8):
    """Compute m0, m2, n0, n2 from parameters."""
    p = _coeffs_from_params(a0, a4, a8)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    m0 = m2 = n0 = n2 = 0.0
    for i in range(7):
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            idx0 = i + j
            idx2 = i + j + 2
            if idx0 < 25:
                m0 += pij * _I[idx0]
                n0 += qij * _I[idx0]
            if idx2 < 25:
                m2 += pij * _I[idx2]
                n2 += qij * _I[idx2]
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    return m0, m2, n0, n2

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound sqrt(vf * vF)."""
    m0, m2, n0, n2 = _compute_m_n(a0, a4, a8)
    if m0 <= 1e-12 or n0 <= 1e-12 or m2 <= 1e-12 or n2 <= 1e-12:
        return 1e18
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    if not np.isfinite(bound):
        return 1e18
    return bound

def _compute_bound_and_analytic_grad(a0, a4, a8):
    """
    Compute the bound and its analytic gradient w.r.t (a0, a4, a8).
    """
    p = _coeffs_from_params(a0, a4, a8)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    m0, m2, n0, n2 = _compute_m_n(a0, a4, a8)
    
    if m0 <= 1e-12 or n0 <= 1e-12 or m2 <= 1e-12 or n2 <= 1e-12:
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Analytic gradient computation
    # dp/da for each parameter
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    dp_da = np.zeros((7, 3))
    # p[0] = 0 always
    # p[1] = -6a4 - 420a8 - 7560a12
    dp_da[1, 0] = -7560 * (-1/945)
    dp_da[1, 1] = -6 + (-7560 * (-3/945))
    dp_da[1, 2] = -420 + (-7560 * (-105/945))
    # p[2] = a4 + 210a8 + 13860a12
    dp_da[2, 0] = 13860 * (-1/945)
    dp_da[2, 1] = 1 + (13860 * (-3/945))
    dp_da[2, 2] = 210 + (13860 * (-105/945))
    # p[3] = -28a8 - 6930a12
    dp_da[3, 0] = -6930 * (-1/945)
    dp_da[3, 1] = -6930 * (-3/945)
    dp_da[3, 2] = -28 + (-6930 * (-105/945))
    # p[4] = a8 + 1155a12
    dp_da[4, 0] = 1155 * (-1/945)
    dp_da[4, 1] = 1155 * (-3/945)
    dp_da[4, 2] = 1 + (1155 * (-105/945))
    # p[5] = -66a12
    dp_da[5, 0] = -66 * (-1/945)
    dp_da[5, 1] = -66 * (-3/945)
    dp_da[5, 2] = -66 * (-105/945)
    # p[6] = a12
    dp_da[6, 0] = -1/945
    dp_da[6, 1] = -3/945
    dp_da[6, 2] = -105/945
    
    grad = np.zeros(3)
    
    for idx in range(3):
        dp_dac = dp_da[:, idx]
        
        # Compute dm0, dm2
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            for j in range(7):
                dmij = dp_dac[i] * p[j] + p[i] * dp_dac[j]
                if dmij == 0.0:
                    continue
                idx0 = i + j
                idx2 = i + j + 2
                if idx0 < 25:
                    dm0 += dmij * _I[idx0]
                if idx2 < 25:
                    dm2 += dmij * _I[idx2]
                    
        # Compute dn0, dn2
        dq_dac = dp_dac * signs
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            for j in range(7):
                dnij = dq_dac[i] * q[j] + q[i] * dq_dac[j]
                if dnij == 0.0:
                    continue
                idx0 = i + j
                idx2 = i + j + 2
                if idx0 < 25:
                    dn0 += dnij * _I[idx0]
                if idx2 < 25:
                    dn2 += dnij * _I[idx2]
                    
        dn0 *= 2.0 * np.pi
        dn2 *= 2.0 * np.pi
        
        d_vf = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        d_vF = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[idx] = (vF * d_vf + vf * d_vF) / (2.0 * bound)
        
    return bound, grad

def _objective_2d(params):
    """Objective for 2D params (a4, a8) with a0=1."""
    return _compute_bound(1.0, params[0], params[1])

def _objective_2d_grad(params):
    """Objective and gradient for 2D params."""
    bound, grad3 = _compute_bound_and_analytic_grad(1.0, params[0], params[1])
    return bound, np.array([grad3[1], grad3[2]])

def _remez_exchange_step(params, best_val, n_samples=50):
    """
    Remez-like exchange: sample the bound landscape, find where it's
    equioscillating, and adjust parameters to reduce the maximum deviation.
    """
    a4, a8 = params
    # Sample a grid around current point
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    samples = []
    values = []
    for _ in range(n_samples):
        da4 = rng.normal(0, 0.01)
        da8 = rng.normal(0, 0.001)
        s = np.array([a4 + da4, a8 + da8])
        try:
            v = _compute_bound(1.0, s[0], s[1])
            if np.isfinite(v) and v < 10.0:
                samples.append(s)
                values.append(v)
        except:
            pass
    
    if len(values) < 5:
        return params, best_val
    
    values = np.array(values)
    samples = np.array(samples)
    
    # Find the minimum and maximum in the sampled region
    vmin = np.min(values)
    vmax = np.max(values)
    
    if vmax - vmin < 1e-6:
        return params, best_val
    
    # The Remez idea: we want to equiosculate the error
    # Find points where the bound is at local minima and maxima
    # and adjust to reduce the spread
    
    # Simple heuristic: move towards the direction that reduces the spread
    # by averaging the gradients at min and max points
    idx_min = np.argmin(values)
    idx_max = np.argmax(values)
    
    p_min = samples[idx_min]
    p_max = samples[idx_max]
    
    # Move towards the midpoint in parameter space, weighted by how far
    # each is from the current best
    mid = (p_min + p_max) / 2.0
    
    # Blend: 70% current, 30% midpoint
    new_params = 0.7 * params + 0.3 * mid
    
    # Evaluate
    new_val = _compute_bound(1.0, new_params[0], new_params[1])
    
    if new_val < best_val:
        return new_params, new_val
    
    # Try moving towards the minimum point
    new_params2 = 0.5 * params + 0.5 * p_min
    new_val2 = _compute_bound(1.0, new_params2[0], new_params2[1])
    
    if new_val2 < best_val:
        return new_params2, new_val2
    
    return params, best_val

def construct():
    best_val = np.inf
    best_params = np.array([0.0, 0.0])
    
    # Phase 1: Differential Evolution Global Search
    bounds = [(-1.0, 1.0), (-0.05, 0.05)]
    
    if time.time() < _DEADLINE - 120.0:
        try:
            de_res = differential_evolution(
                _objective_2d, bounds,
                maxiter=300, popsize=20,
                tol=1e-12, atol=1e-12,
                seed=42,
                polish=False
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with Remez exchange
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [best_params.copy()]
    
    # Around the known good region
    for _ in range(15):
        u = rng.normal(-0.035, 0.1)
        v = rng.normal(-0.0003, 0.005)
        starts.append(np.array([u, v]))
    
    # Wider exploration
    for _ in range(25):
        u = rng.uniform(-0.8, 0.8)
        v = rng.uniform(-0.03, 0.03)
        starts.append(np.array([u, v]))
    
    # Structured grid
    for u in np.linspace(-0.5, 0.5, 7):
        for v in np.linspace(-0.02, 0.02, 5):
            starts.append(np.array([u, v]))
    
    # Evaluate all starts and keep top performers
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 60.0:
            break
        try:
            val = _compute_bound(1.0, s[0], s[1])
            if np.isfinite(val):
                scored.append((val, s.copy()))
        except:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Polish top starts with L-BFGS-B
    top_n = min(20, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 30.0:
            break
        val, s = scored[i]
        try:
            res = minimize(
                _objective_2d_grad, s, method='L-BFGS-B',
                jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 3: Remez exchange iterations
    for iteration in range(10):
        if time.time() > _DEADLINE - 20.0:
            break
        best_params, best_val = _remez_exchange_step(best_params, best_val, n_samples=80)
        
        # Polish after Remez step
        try:
            res = minimize(
                _objective_2d_grad, best_params, method='L-BFGS-B',
                jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(
                _objective_2d_grad, best_params, method='L-BFGS-B',
                jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Convert back to (a0, a4, a8)
    a0 = 1.0
    a4 = best_params[0]
    a8 = best_params[1]
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
