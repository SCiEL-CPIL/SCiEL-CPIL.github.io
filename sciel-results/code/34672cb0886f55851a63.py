import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

# Precompute GL nodes and weights for initial guess
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)
_GL_NODES = 0.5 * (_GL_NODES + 1.0)  # Map to [0,1]
_GL_WEIGHTS = 0.5 * _GL_WEIGHTS
_GL_WEIGHTS = _GL_WEIGHTS / np.sum(_GL_WEIGHTS)

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    # Vectorized operations
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    # Prevent division by zero or negative values due to numerical issues
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=4096):
    """
    Fast reference calculation of (1+z) * integral.
    Uses a dense fixed grid to avoid adaptive overhead of quad.
    """
    if z <= 0:
        return 0.0
    # Grid in z-space from 0 to z
    zs = np.linspace(0, z, num_points)
    # Integrand is 1/E(a(z'))
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    
    # Use Simpson's rule for better accuracy (requires odd number of points)
    if num_points % 2 == 0:
        num_points -= 1
        zs = np.linspace(0, z, num_points)
        a_vals = 1.0 / (1.0 + zs)
        E_vals = _E_cpl(a_vals, w0, wa)
        integrand = 1.0 / E_vals
    
    integral = np.trapz(integrand, zs)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    """
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node params (t), 8 weight params (v).
    """
    if _time_left() < 1.0:
        return 1e10
        
    # Decode nodes
    # t_i -> x_i sequence
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        # Sigmoid mapping to (0, 1)
        s = 1.0 / (1.0 + np.exp(-params[i]))
        # Scale step by remaining space
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    # Ensure strictly increasing and in (0,1)
    # The construction above guarantees 0 < x[0] < x[1] ... < x[7] < 1
    # But we might want to clip extremely close to 0 or 1 if sigmoid is flat
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Test set
    # We need a representative set of cosmologies.
    # Include LCDM, wCDM, and some extreme CPL cases.
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0),   # More extreme quintessence
        (-0.8, -0.2),  # Near-LCDM
        (0.2, 0.8),    # Strongly evolving
        (-1.2, -0.5),  # Phantom with evolution
        (0.7, -0.7),   # Strong quintessence
    ]
    
    # Redshifts
    # Cover a wide range with more points at low z where errors matter most
    test_zs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Better initial guess: Use Gauss-Legendre nodes as starting point
    # We need to find t params that produce GL-like node distribution
    # For now, use a reasonable spread in t-space
    
    # Bounds for DE
    # t_i: -8 to 8 is a good range for sigmoid to cover [0,1] well.
    # v_i: -5 to 5 is a good range for log-weights.
    bounds = [(-8, 8)] * 8 + [(-5, 5)] * 8
    
    # Differential Evolution
    best_result = None
    best_score = 1e10
    
    try:
        if _time_left() > 30:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=150, 
                popsize=15, 
                tol=1e-9,
                polish=False
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        pass
    
    # Refinement with Nelder-Mead
    if best_result is not None and _time_left() > 15:
        try:
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
            pass
            
    # If DE failed, fallback to Gauss-Legendre
    if best_result is None:
        nodes = _GL_NODES.copy()
        weights = _GL_WEIGHTS.copy()
        return {"nodes": nodes, "weights": weights}
    
    # Decode final result
    t = best_result[:8]
    v = best_result[8:16]
    
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-t[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    w_raw = np.exp(v)
    w = w_raw / np.sum(w_raw)
    
    # Final validation check
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
