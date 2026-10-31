"""Uncertainty inequality optimization via 2D global search and analytic gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

# Set a deadline slightly before the 210s limit to be safe
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
    and a12 is chosen such that P(0)=0.
    H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945.
    P(0) = a0 + 3a4 + 105a8 + 945a12 = 0 => a12 = -(a0 + 3a4 + 105a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p = np.zeros(7)
    # p[0] (x^0 term)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # Should be 0
    # p[1] (x^2 term)
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    # p[2] (x^4 term)
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    # p[3] (x^6 term)
    p[3] = -28.0 * a8 - 6930.0 * a12
    # p[4] (x^8 term)
    p[4] = a8 + 1155.0 * a12
    # p[5] (x^10 term)
    p[5] = -66.0 * a12
    # p[6] (x^12 term)
    p[6] = a12
    return p

def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the bound sqrt(vf * vF) and its gradient w.r.t (a0, a4, a8).
    vf = m2/m0, vF = n2/n0.
    m_k = sum_{i,j} p_i p_j I[k+i+j]
    n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    q_i = (-1)^i p_i
    """
    p = _coeffs_from_params(a0, a4, a8)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    # Compute m0, m2 and n0, n2
    # m_k = sum_{i=0}^6 sum_{j=0}^6 p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    # Precompute sums for efficiency
    # m_k = sum_i p_i (sum_j p_j I[k+i+j])
    # We can just do the double loop, it's small (7x7)
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
    
    if m0 <= 1e-12 or n0 <= 1e-12 or m2 <= 1e-12 or n2 <= 1e-12:
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Analytic Gradient
    # bound = (m2*m2*n2*n2)^(1/4) ... wait, bound = sqrt(m2/m0 * n2/n0)
    # Let R = (m2/m0)*(n2/n0). bound = sqrt(R).
    # d(bound)/da = (1/(2*bound)) * dR/da
    # dR/da = R * (d(m2/m0)/da + d(n2/n0)/da)
    # d(m2/m0)/da = (dm2/da * m0 - m2 * dm0/da) / m0^2
    # Similarly for n.
    
    # We need dm0, dm2, dn0, dn2 for each parameter a0, a4, a8.
    # p is linear in a. dp/da is constant vectors.
    
    # dp/da0, dp/da4, dp/da8
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945
    # da12/da4 = -3/945
    # da12/da8 = -105/945
    
    # p[0] = 0 (always)
    # p[1] = -6a4 - 420a8 - 7560a12
    # p[2] = a4 + 210a8 + 13860a12
    # p[3] = -28a8 - 6930a12
    # p[4] = a8 + 1155a12
    # p[5] = -66a12
    # p[6] = a12
    
    # Derivatives of p w.r.t a0, a4, a8
    dp_da0 = np.zeros(7)
    dp_da4 = np.zeros(7)
    dp_da8 = np.zeros(7)
    
    # Coeffs of a12 in p:
    # p1: -7560, p2: 13860, p3: -6930, p4: 1155, p5: -66, p6: 1
    
    # da12/da0 = -1/945
    dp_da0[1] = -7560 * (-1/945)
    dp_da0[2] = 13860 * (-1/945)
    dp_da0[3] = -6930 * (-1/945)
    dp_da0[4] = 1155 * (-1/945)
    dp_da0[5] = -66 * (-1/945)
    dp_da0[6] = 1 * (-1/945)
    
    # da12/da4 = -3/945
    dp_da4[1] = -6 + (-7560 * (-3/945))
    dp_da4[2] = 1 + (13860 * (-3/945))
    dp_da4[3] = (-6930 * (-3/945))
    dp_da4[4] = (1155 * (-3/945))
    dp_da4[5] = (-66 * (-3/945))
    dp_da4[6] = 1 * (-3/945)
    
    # da12/da8 = -105/945
    dp_da8[1] = -420 + (-7560 * (-105/945))
    dp_da8[2] = 210 + (13860 * (-105/945))
    dp_da8[3] = -28 + (-6930 * (-105/945))
    dp_da8[4] = 1 + (1155 * (-105/945))
    dp_da8[5] = (-66 * (-105/945))
    dp_da8[6] = 1 * (-105/945)
    
    grad = np.zeros(3)
    R = vf * vF
    
    for idx, dp_dac in enumerate([dp_da0, dp_da4, dp_da8]):
        # Compute dm0, dm2
        dm0 = 0.0
        dm2 = 0.0
        # dm_k = sum_{i,j} (dp_i p_j + p_i dp_j) I[k+i+j]
        # = 2 * sum_i dp_i (sum_j p_j I[k+i+j]) if symmetric? No, just double sum.
        for i in range(7):
            dpi = dp_dac[i]
            pi = p[i]
            if dpi == 0.0 and pi == 0.0:
                continue
            for j in range(7):
                pij = p[i] * p[j]
                dmij = dpi * p[j] + p[i] * dp_dac[j]
                if dmij == 0.0:
                    continue
                idx0 = i + j
                idx2 = i + j + 2
                if idx0 < 25:
                    dm0 += dmij * _I[idx0]
                if idx2 < 25:
                    dm2 += dmij * _I[idx2]
                    
        # Compute dn0, dn2
        # q_i = s_i p_i, so dq_i = s_i dp_i
        dq_dac = dp_dac * signs
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq_dac[i]
            qi = q[i]
            if dqi == 0.0 and qi == 0.0:
                continue
            for j in range(7):
                qij = q[i] * q[j]
                dnij = dqi * q[j] + q[i] * dq_dac[j]
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
        
        # d(bound)/da = (1/(2*bound)) * (vF * d_vf + vf * d_vF)
        grad[idx] = (vF * d_vf + vf * d_vF) / (2.0 * bound)
        
    return bound, grad

def _objective_2d(params):
    """Objective function for 2D parameters (u, v) where a0=1, a4=u, a8=v."""
    a0, a4, a8 = 1.0, params[0], params[1]
    bound, _ = _compute_bound_and_grad(a0, a4, a8)
    return bound

def _objective_2d_grad(params):
    """Objective and gradient for 2D parameters."""
    a0, a4, a8 = 1.0, params[0], params[1]
    bound, grad = _compute_bound_and_grad(a0, a4, a8)
    # grad is w.r.t (a0, a4, a8). We need w.r.t (a4, a8).
    # a0 is fixed, so we only take grad[1] and grad[2].
    # Note: The gradient w.r.t a4 and a8 in the full space is what we want.
    # Since a0 is constant, the chain rule doesn't change the components for a4, a8.
    return bound, np.array([grad[1], grad[2]])

def construct():
    best_val = np.inf
    best_params = np.array([0.0, 0.0]) # (a4, a8) with a0=1
    
    # Phase 1: Differential Evolution Global Search
    # Bounds for u=a4/a0 and v=a8/a0
    # Based on previous good solution: a0=-1.13, a4=0.04, a8=0.0003
    # u ~ -0.035, v ~ -0.00027
    # Let's search a wider range to be safe.
    bounds = [(-1.0, 1.0), (-0.05, 0.05)]
    
    if time.time() < _DEADLINE - 100.0:
        try:
            de_res = differential_evolution(
                _objective_2d, bounds, 
                maxiter=200, popsize=15, 
                tol=1e-12, atol=1e-12, 
                seed=42, 
                polish=False # We will polish manually
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x
        except Exception as e:
            pass
            
    # Phase 2: L-BFGS-B Polish from DE result and a few random starts
    starts = [best_params]
    # Add some random starts in the promising region
    rng = np.random.default_rng(42)
    for _ in range(10):
        u = rng.uniform(-0.5, 0.5)
        v = rng.uniform(-0.02, 0.02)
        starts.append(np.array([u, v]))
        
    # Also add the known good point from the previous iteration (scaled to a0=1)
    # a0=-1.13, a4=0.04, a8=0.000308642
    # u = 0.04/-1.13 = -0.0354, v = 0.000308642/-1.13 = -0.000273
    starts.append(np.array([-0.0354, -0.000273]))
    
    for start in starts:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            res = minimize(
                _objective_2d_grad, start, method='L-BFGS-B',
                jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Phase 3: Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(
                _objective_2d_grad, best_params, method='L-BFGS-B',
                jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Convert back to (a0, a4, a8)
    # We fixed a0=1. The problem is scale invariant.
    # The evaluator expects 3 coefficients.
    # Returning (1, a4, a8) is a valid representative.
    a0 = 1.0
    a4 = best_params[0]
    a8 = best_params[1]
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
