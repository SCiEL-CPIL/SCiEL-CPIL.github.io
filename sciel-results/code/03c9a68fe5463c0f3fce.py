"""Uncertainty inequality optimization: analytic-gradient L-BFGS-B with multi-start.

Structural change: replace finite-difference gradients with exact analytic gradients,
and replace the heavy differential-evolution global phase with a dense multi-start
local search seeded from a Sobol quasi-random grid. The analytic gradient is ~100x
faster per L-BFGS-B step, allowing many more starts and deeper polishing within
the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

_DEADLINE = time.time() + 192  # 18s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12."""
    I = np.zeros(13)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 13):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()
_sqrt_pi = np.sqrt(np.pi)
_two_pi = 2.0 * np.pi


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


def _compute_bound_and_grad(params):
    """
    Compute the uncertainty product sqrt(vf * vF) and its analytic gradient
    with respect to (a0, a4, a8).
    
    vf = m2/m0, vF = n2/n0, bound = sqrt(vf*vF)
    d(bound)/da_j = (1/(2*bound)) * (vF * d(vf)/da_j + vf * d(vF)/da_j)
    d(vf)/da_j = (m2' * m0 - m2 * m0') / m0^2
    d(vF)/da_j = (n2' * n0 - n2 * n0') / n0^2
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Precompute pairwise products and their index sums
    m = np.zeros(3)  # m[0]=m0, m[1]=m2, m[2]=m4 (we need m0, m2)
    n = np.zeros(3)
    
    # Actually we only need m0, m2, n0, n2
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
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
            idx_m0 = i + j
            idx_m2 = 2 + i + j
            if idx_m0 < 13:
                m0 += pij * _I[idx_m0]
            if idx_m2 < 13:
                m2 += pij * _I[idx_m2]
            n0 += qij * _I[idx_m0]
            n2 += qij * _I[idx_m2]
    
    n0 *= _two_pi
    n2 *= _two_pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Gradient computation
    # dp/da0, dp/da4, dp/da8
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    # p[0] = a0+3a4+105a8+945*a12 = a0+3a4+105a8 - (a0+3a4+105a8) = 0  (always!)
    # Wait, that's identically zero. Let me recheck.
    # p[0] = a0 + 3*a4 + 105*a8 + 945*a12
    # a12 = -(a0+3a4+105a8)/945
    # 945*a12 = -(a0+3a4+105a8)
    # p[0] = a0+3a4+105a8 - (a0+3a4+105a8) = 0
    # So p[0] is always 0! This means the constant term vanishes.
    
    # Let me recompute p more carefully:
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12  (physicist's convention? or probabilist's?)
    # Actually, let me use the standard Hermite polynomials.
    # The task says H0, H4, H8, H12. Let's use physicist's Hermite:
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112
    # H12 = 4096x^12 - 46080x^10 + 241920x^8 - 552960x^6 + 622080x^4 - 290304x^2 + 39916... 
    
    # Wait, I need to be more careful. Let me use the recursive definition.
    # Actually, the current code has specific coefficients. Let me just compute
    # dp/da_j analytically from the formulas.
    
    # p[0] = 0 (as shown above)
    # p[1] = -6*a4 - 420*a8 - 7560*a12
    # p[2] = a4 + 210*a8 + 13860*a12
    # p[3] = -28*a8 - 6930*a12
    # p[4] = a8 + 1155*a12
    # p[5] = -66*a12
    # p[6] = a12
    
    # dp_k/da0:
    # dp0/da0 = 0 (p0 is identically 0)
    # dp1/da0 = -7560 * da12/da0 = -7560 * (-1/945) = 8
    # dp2/da0 = 13860 * da12/da0 = 13860 * (-1/945) = -14.666... = -44/3
    # dp3/da0 = -6930 * da12/da0 = -6930 * (-1/945) = 7.333... = 22/3
    # dp4/da0 = 1155 * da12/da0 = 1155 * (-1/945) = -1.222... = -11/9
    # dp5/da0 = -66 * da12/da0 = -66 * (-1/945) = 66/945 = 22/315
    # dp6/da0 = da12/da0 = -1/945
    
    # dp_k/da4:
    # dp1/da4 = -6 - 7560 * (-3/945) = -6 + 24 = 18
    # dp2/da4 = 1 + 13860 * (-3/945) = 1 - 44 = -43
    # dp3/da4 = -6930 * (-3/945) = 22
    # dp4/da4 = 1155 * (-3/945) = -3.666... = -11/3
    # dp5/da4 = -66 * (-3/945) = 198/945 = 22/105
    # dp6/da4 = -3/945 = -1/315
    
    # dp_k/da8:
    # dp1/da8 = -420 - 7560 * (-105/945) = -420 + 840 = 420
    # dp2/da8 = 210 + 13860 * (-105/945) = 210 - 1540 = -1330
    # dp3/da8 = -28 - 6930 * (-105/945) = -28 + 770 = 742
    # dp4/da8 = 1 + 1155 * (-105/945) = 1 - 128.333... = -127.333... = -382/3
    # dp5/da8 = -66 * (-105/945) = 6930/945 = 7.333... = 22/3
    # dp6/da8 = -105/945 = -1/9
    
    dp = np.zeros((7, 3))
    # da0
    dp[1, 0] = 8.0
    dp[2, 0] = -44.0/3.0
    dp[3, 0] = 22.0/3.0
    dp[4, 0] = -11.0/9.0
    dp[5, 0] = 22.0/315.0
    dp[6, 0] = -1.0/945.0
    # da4
    dp[1, 1] = 18.0
    dp[2, 1] = -43.0
    dp[3, 1] = 22.0
    dp[4, 1] = -11.0/3.0
    dp[5, 1] = 22.0/105.0
    dp[6, 1] = -1.0/315.0
    # da8
    dp[1, 2] = 420.0
    dp[2, 2] = -1330.0
    dp[3, 2] = 742.0
    dp[4, 2] = -382.0/3.0
    dp[5, 2] = 22.0/3.0
    dp[6, 2] = -1.0/9.0
    
    # dq_k/da_j: q_k = p_k * (-1)^k, so dq_k/da_j = dp_k/da_j * (-1)^k
    sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    dq = dp * sign[:, np.newaxis]
    
    # Now compute dm0/da_j, dm2/da_j, dn0/da_j, dn2/da_j
    # m0 = sum_{i,j} p_i p_j I_{i+j}
    # dm0/da_j = sum_{i,k} (dp_i/da_j p_k + p_i dp_k/da_j) I_{i+k}
    # = 2 * sum_{i,k} dp_i/da_j p_k I_{i+k}  (by symmetry)
    
    dm0_da = np.zeros(3)
    dm2_da = np.zeros(3)
    dn0_da = np.zeros(3)
    dn2_da = np.zeros(3)
    
    for j in range(3):
        dm0_da[j] = 0.0
        dm2_da[j] = 0.0
        dn0_da[j] = 0.0
        dn2_da[j] = 0.0
        for i in range(7):
            dpi = dp[i, j]
            dq_i = dq[i, j]
            if dpi == 0.0 and dq_i == 0.0:
                continue
            for k in range(7):
                pk = p[k]
                qk = q[k]
                if pk == 0.0 and qk == 0.0:
                    continue
                idx0 = i + k
                idx2 = 2 + i + k
                if idx0 < 13:
                    dm0_da[j] += 2.0 * dpi * pk * _I[idx0]
                    dn0_da[j] += 2.0 * dq_i * qk * _I[idx0]
                if idx2 < 13:
                    dm2_da[j] += 2.0 * dpi * pk * _I[idx2]
                    dn2_da[j] += 2.0 * dq_i * qk * _I[idx2]
        dn0_da[j] *= _two_pi
        dn2_da[j] *= _two_pi
    
    # d(vf)/da_j = (dm2_da[j]*m0 - m2*dm0_da[j]) / m0^2
    # d(vF)/da_j = (dn2_da[j]*n0 - n2*dn0_da[j]) / n0^2
    # d(bound)/da_j = (1/(2*bound)) * (vF * d(vf)/da_j + vf * d(vF)/da_j)
    
    m0_sq = m0 * m0
    n0_sq = n0 * n0
    
    d_vf = (dm2_da * m0 - m2 * dm0_da) / m0_sq
    d_vF = (dn2_da * n0 - n2 * dn0_da) / n0_sq
    
    grad = (vF * d_vf + vf * d_vF) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val


def _objective_grad(params):
    return _compute_bound_and_grad(params)


def _local_optimize(x0, maxiter=20000):
    """Run L-BFGS-B local optimization from x0 with analytic gradient."""
    try:
        res = minimize(_objective_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16})
        if res.fun < 1e17 and np.isfinite(res.fun):
            return res.x, res.fun
    except Exception:
        pass
    return x0, _objective(x0)


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(12345)
    t0 = time.time()
    
    # Generate a dense Sobol quasi-random grid for multi-start
    # We want many starting points in a reasonable bounding box
    n_points = 2000
    sampler = qmc.Sobol(d=3, scramble=True, seed=42)
    samples = sampler.random(n_points)
    
    # Map to reasonable bounds
    # Based on the known good solution around (-1.13, 0.04, 0.000308642)
    bounds_low = np.array([-3.0, -0.3, -0.02])
    bounds_high = np.array([1.0, 0.3, 0.02])
    starts = bounds_low + samples * (bounds_high - bounds_low)
    
    # Add some additional starts near the known good solution
    for scale in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        for _ in range(20):
            starts = np.vstack([starts, best_params + rng.normal(0, scale, 3)])
    
    # Also add some random starts in a wider box
    for _ in range(200):
        starts = np.vstack([starts, rng.uniform([-5, -1, -0.1], [2, 1, 0.1])])
    
    # Evaluate all starts quickly
    start_vals = np.array([_objective(s) for s in starts])
    
    # Sort by value and take top candidates for local optimization
    order = np.argsort(start_vals)
    top_n = min(80, len(order))
    top_indices = order[:top_n]
    
    # Local optimization from top starts
    for idx in top_indices:
        if time.time() - t0 > _DEADLINE - 20:
            break
        x_opt, f_opt = _local_optimize(starts[idx])
        if f_opt < best_val:
            best_val = f_opt
            best_params = x_opt.copy()
    
    # Phase 2: Basin hopping around the best point
    current_best = best_params.copy()
    current_val = best_val
    
    scales = [0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for scale in scales:
        if time.time() - t0 > _DEADLINE - 15:
            break
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 15:
                break
            x_pert = current_best + rng.normal(0, scale, 3)
            x_opt, f_opt = _local_optimize(x_pert)
            if f_opt < current_val:
                current_val = f_opt
                current_best = x_opt.copy()
                if f_opt < best_val:
                    best_val = f_opt
                    best_params = x_opt.copy()
    
    # Phase 3: Intensive polishing from multiple nearby points
    if time.time() - t0 < _DEADLINE - 10:
        polish_starts = [best_params.copy()]
        for scale in [0.01, 0.005, 0.001, 0.0005]:
            for _ in range(5):
                polish_starts.append(best_params + rng.normal(0, scale, 3))
        
        for ps in polish_starts:
            if time.time() - t0 > _DEADLINE - 10:
                break
            x_opt, f_opt = _local_optimize(ps, maxiter=30000)
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
    
    # Phase 4: Nelder-Mead for a different local optimizer perspective
    if time.time() - t0 < _DEADLINE - 5:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 30000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise L-BFGS-B polishing
    if time.time() - t0 < _DEADLINE:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
