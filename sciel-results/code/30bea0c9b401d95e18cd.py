"""
Optimized Uncertainty Bound Minimizer
Approach: Differential Evolution (Global) + Nelder-Mead (Local) with L2-Normalized Parameters.
This avoids scale invariance issues and uses distinct optimization algorithms from the seed program.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time

_DEADLINE = time.time() + 160

# Pre-compute Gauss-Hermite nodes and weights
# Same as before, exact for the polynomial degree involved
_N_QUAD = 15
_x_nodes, _w_weights = roots_hermite(_N_QUAD)

def _make_g_coefficients(a0, a4, a8):
    """
    Compute polynomial coefficients P(x) = sum p_k x^{2k}.
    a12 is fixed to enforce g(0) = 0 => P(0) = 0 => p0 = 0.
    """
    # p0 = a0 + 3 a4 + 105 a8 + 945 a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p0 = 0.0 # By construction
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    
    return np.array([p0, p1, p2, p3, p4, p5, p6])

def _compute_moments(p, nodes, weights):
    """
    Compute m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1
    """
    x2 = nodes**2
    P_vals = np.zeros_like(nodes)
    # P(x) is even, so we only need even powers.
    # p0 is 0, so we start from i=1
    for i in range(1, 7):
        P_vals += p[i] * (x2 ** i)
        
    P_sq = P_vals * P_vals
    
    m0 = np.sum(weights * P_sq)
    m1 = np.sum(weights * (nodes**2) * P_sq)
    
    return m0, m1

def _compute_fourier_moments(p, nodes, weights):
    """
    Compute n_k = 2*pi * integral xi^{2k} |Q(xi)|^2 e^{-xi^2} dxi for k=0,1
    Q(xi) coefficients q_i = p_i * (-1)^i
    """
    q = p * ((-1.0) ** np.arange(7))
    
    x2 = nodes**2
    Q_vals = np.zeros_like(nodes)
    for i in range(1, 7):
        Q_vals += q[i] * (x2 ** i)
        
    Q_sq = Q_vals * Q_vals
    
    n0 = 2.0 * np.pi * np.sum(weights * Q_sq)
    n1 = 2.0 * np.pi * np.sum(weights * (nodes**2) * Q_sq)
    
    return n0, n1

def _bound_value(params):
    """
    Compute the uncertainty bound.
    Normalizes input parameters to unit L2 norm to handle scale invariance.
    """
    # Normalize
    norm = np.linalg.norm(params)
    if norm < 1e-12:
        return 1e18
        
    a_norm = params / norm
    a0, a4, a8 = a_norm[0], a_norm[1], a_norm[2]
    
    p = _make_g_coefficients(a0, a4, a8)
    
    m0, m1 = _compute_moments(p, _x_nodes, _w_weights)
    n0, n1 = _compute_fourier_moments(p, _x_nodes, _w_weights)
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0) or not np.isfinite(m1) or not np.isfinite(n1):
        return 1e18
        
    if m1 < 0 or n1 < 0:
        return 1e18
        
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def construct():
    """
    Find optimal Hermite coefficients.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Known good seed from previous iteration (un-normalized)
    # We will evaluate this and the optimized results
    seed_raw = np.array([-1.13, 0.04, 0.000308642])
    
    best_val = 1e18
    best_params = seed_raw.copy()
    
    # Evaluate the seed
    seed_val = _bound_value(seed_raw)
    if seed_val < best_val:
        best_val = seed_val
        best_params = seed_raw.copy()
        
    # Global Search using Differential Evolution
    if time.time() < _DEADLINE - 30:
        try:
            # Bounds for the normalized parameters (they are on the sphere, but DE works in a box)
            # Since we normalize inside, the box can be small, e.g., [-1, 1]
            bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
            
            res_de = differential_evolution(
                _bound_value, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15,
                tol=1e-10,
                polish=True, # DE does a local polish at the end
                x0=seed_raw / np.linalg.norm(seed_raw) # Start near known good point
            )
            
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass

    # Local Polish using Nelder-Mead from the best found point
    if time.time() < _DEADLINE - 5:
        try:
            # Nelder-Mead is good for low-dimensional, smooth but potentially tricky landscapes
            res_nm = minimize(
                _bound_value, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
            
    # Ensure we return finite values
    if not np.all(np.isfinite(best_params)):
        return seed_raw
        
    return np.array(best_params)
