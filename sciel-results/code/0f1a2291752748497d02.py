"""Uncertainty inequality optimization via generalized eigenvalue in Hermite basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

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
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # should be ~0
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
                if idx < 25:
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


def _bound_and_analytic_grad(params):
    """
    Compute bound and analytic gradient using the quotient rule.
    bound = sqrt((m2/m0) * (n2/n0))
    Let R = (m2*n2)/(m0*n0), bound = sqrt(R)
    d(bound)/da = (1/(2*bound)) * dR/da
    dR/da = R * (d(m2/m0)/da + d(n2/n0)/da)
    d(m2/m0)/da = (dm2/da * m0 - m2 * dm0/da) / m0^2
    Similarly for n.
    """
    a0, a4, a8 = params
    p = _coeffs_from_params(a0, a4, a8)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute m0, m2, n0, n2 and their derivatives w.r.t. a0, a4, a8
    # p depends on (a0, a4, a8) linearly via a12 = -(a0+3a4+105a8)/945
    
    # dp/da0, dp/da4, dp/da8
    dp_da0 = np.zeros(7)
    dp_da4 = np.zeros(7)
    dp_da8 = np.zeros(7)
    
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    # p[0] = a0 + 3a4 + 105a8 + 945*a12 = 0 (by construction)
    # p[1] = -6a4 - 420a8 - 7560*a12
    # p[2] = a4 + 210a8 + 13860*a12
    # p[3] = -28a8 - 6930*a12
    # p[4] = a8 + 1155*a12
    # p[5] = -66*a12
    # p[6] = a12
    
    # dp[0]/da0 = 1 + 945*(-1/945) = 0
    # dp[0]/da4 = 3 + 945*(-3/945) = 0
    # dp[0]/da8 = 105 + 945*(-105/945) = 0
    
    # dp[1]/da0 = -7560*(-1/945) = 8
    # dp[1]/da4 = -6 - 7560*(-3/945) = -6 + 24 = 18
    # dp[1]/da8 = -420 - 7560*(-105/945) = -420 + 840 = 420
    
    # dp[2]/da0 = 13860*(-1/945) = -14.666... = -44/3
    # dp[2]/da4 = 1 + 13860*(-3/945) = 1 - 44 = -43
    # dp[2]/da8 = 210 + 13860*(-105/945) = 210 - 1540 = -1330
    
    # dp[3]/da0 = -6930*(-1/945) = 7.333... = 22/3
    # dp[3]/da4 = -6930*(-3/945) = 22
    # dp[3]/da8 = -28 - 6930*(-105/945) = -28 + 770 = 742
    
    # dp[4]/da0 = 1155*(-1/945) = -1.222... = -11/9
    # dp[4]/da4 = 1155*(-3/945) = -3.666... = -11/3
    # dp[4]/da8 = 1 + 1155*(-105/945) = 1 - 128.333... = -127.333... = -382/3
    
    # dp[5]/da0 = -66*(-1/945) = 66/945 = 22/315
    # dp[5]/da4 = -66*(-3/945) = 198/945 = 22/105
    # dp[5]/da8 = -66*(-105/945) = 6930/945 = 22/3
    
    # dp[6]/da0 = -1/945
    # dp[6]/da4 = -3/945 = -1/315
    # dp[6]/da8 = -105/945 = -1/9
    
    dp_da0[1] = 8.0
    dp_da0[2] = -44.0/3.0
    dp_da0[3] = 22.0/3.0
    dp_da0[4] = -11.0/9.0
    dp_da0[5] = 22.0/315.0
    dp_da0[6] = -1.0/945.0
    
    dp_da4[1] = 18.0
    dp_da4[2] = -43.0
    dp_da4[3] = 22.0
    dp_da4[4] = -11.0/3.0
    dp_da4[5] = 22.0/105.0
    dp_da4[6] = -1.0/315.0
    
    dp_da8[1] = 420.0
    dp_da8[2] = -1330.0
    dp_da8[3] = 742.0
    dp_da8[4] = -382.0/3.0
    dp_da8[5] = 22.0/3.0
    dp_da8[6] = -1.0/9.0
    
    # Compute m_k = sum_{i,j} p_i p_j I[k+i+j]
    # dm_k/da_c = sum_{i,j} (dp_i/da_c * p_j + p_i * dp_j/da_c) * I[k+i+j]
    
    def _compute_m_and_grads(p, dp_dac):
        """Compute m[0..2] and their gradients w.r.t. one parameter."""
        m = np.zeros(3)
        dm = np.zeros(3)
        for i in range(7):
            for j in range(7):
                pij = p[i] * p[j]
                dmij = dp_dac[i] * p[j] + p[i] * dp_dac[j]
                for k in range(3):
                    idx = k + i + j
                    if idx < 25:
                        m[k] += pij * _I[idx]
                        dm[k] += dmij * _I[idx]
        return m, dm
    
    m0, m2, _ = _compute_m_and_grads(p, np.zeros(7))
    
    # For each parameter, compute dm0, dm2, dn0, dn2
    grad = np.zeros(3)
    
    for c, dp_dac in enumerate([dp_da0, dp_da4, dp_da8]):
        m_vals, dm_vals = _compute_m_and_grads(p, dp_dac)
        
        # q = p * sign_array, so dq_dac = dp_dac * sign_array
        sign_arr = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        dq_dac = dp_dac * sign_arr
        
        # Compute n and dn
        n_vals = np.zeros(3)
        dn_vals = np.zeros(3)
        for i in range(7):
            for j in range(7):
                qij = q[i] * q[j]
                dnij = dq_dac[i] * q[j] + q[i] * dq_dac[j]
                for k in range(3):
                    idx = k + i + j
                    if idx < 25:
                        n_vals[k] += qij * _I[idx]
                        dn_vals[k] += dnij * _I[idx]
        n_vals *= 2.0 * np.pi
        dn_vals *= 2.0 * np.pi
        
        m0v, m2v = m_vals[0], m_vals[1]
        n0v, n2v = n_vals[0], n_vals[1]
        dm0v, dm2v = dm_vals[0], dm_vals[1]
        dn0v, dn2v = dn_vals[0], dn_vals[1]
        
        if m0v <= 0 or n0v <= 0 or m2v <= 0 or n2v <= 0:
            grad[c] = 0.0
            continue
        
        # d(m2/m0)/da = (dm2v*m0v - m2v*dm0v) / m0v^2
        d_vf = (dm2v * m0v - m2v * dm0v) / (m0v * m0v)
        d_vF = (dn2v * n0v - n2v * dn0v) / (n0v * n0v)
        
        vf = m2v / m0v
        vF = n2v / n0v
        bound_val = np.sqrt(vf * vF)
        
        if bound_val <= 0 or not np.isfinite(bound_val):
            grad[c] = 0.0
            continue
        
        # d(bound)/da = (1/(2*bound)) * (vF * d_vf + vf * d_vF)
        grad[c] = (vF * d_vf + vf * d_vF) / (2.0 * bound_val)
    
    f0 = _bound_from_params(params)
    return f0, grad


def _objective(params):
    return _bound_from_params(params)


def _objective_grad(params):
    return _bound_and_analytic_grad(params)


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Dense grid search in the 3D parameter space
    # The problem is scale-invariant (bound is homogeneous of degree 0 in a),
    # so we can fix a0 and search over a4/a0 and a8/a0 ratios.
    # But let's do a direct 3D search with good coverage.
    
    # First, evaluate a structured grid
    a0_vals = np.linspace(-3.0, 1.0, 25)
    a4_vals = np.linspace(-0.5, 0.5, 25)
    a8_vals = np.linspace(-0.01, 0.01, 15)
    
    grid_points = []
    for a0 in a0_vals:
        for a4 in a4_vals:
            for a8 in a8_vals:
                grid_points.append(np.array([a0, a4, a8]))
    
    # Add random points
    for _ in range(200):
        pt = np.array([
            rng.uniform(-4.0, 2.0),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.02, 0.02)
        ])
        grid_points.append(pt)
    
    # Add points near known good region
    for _ in range(100):
        pt = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.15, 3)
        grid_points.append(pt)
    
    # Evaluate all grid points
    scored = []
    for pt in grid_points:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val = _bound_from_params(pt)
            if np.isfinite(val):
                scored.append((val, pt))
                if val < best_val:
                    best_val = val
                    best_params = pt.copy()
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Phase 2: L-BFGS-B with analytic gradient from top grid points
    top_n = min(20, len(scored))
    for val, pt in scored[:top_n]:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, pt, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional random restarts with L-BFGS-B
    if time.time() < _DEADLINE - 20:
        for _ in range(30):
            if time.time() > _DEADLINE - 15:
                break
            try:
                # Random start in reasonable range
                start = np.array([
                    rng.uniform(-3.0, 1.0),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.015, 0.015)
                ])
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead polish from best
    if time.time() < _DEADLINE - 10:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20,
                                     'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Try sign-flip variations (the problem may have symmetric minima)
    if time.time() < _DEADLINE - 3:
        try:
            # The bound is invariant under P -> -P, so sign of all coefficients doesn't matter.
            # But individual sign flips may lead to different local minima.
            for flip in [1, 2, 4, 3, 5, 6, 7]:
                if time.time() > _DEADLINE - 2:
                    break
                try:
                    perturbed = best_params.copy()
                    if flip & 1:
                        perturbed[0] *= -1
                    if flip & 2:
                        perturbed[1] *= -1
                    if flip & 4:
                        perturbed[2] *= -1
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
