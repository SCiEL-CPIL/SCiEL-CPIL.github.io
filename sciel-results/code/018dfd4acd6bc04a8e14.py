"""Gauss-Legendre-based 8-node quadrature for luminosity distance with targeted refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL(z, w0, wa):
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * val

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    # Standard Gauss-Legendre on [-1,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map to [0,1]: x_new = 0.5*(x + 1), w_new = 0.5*w
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    # Sort nodes to be strictly increasing
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    return nodes, weights

def _params_to_nodes_weights(params):
    """
    Params 0-7: log-scale shifts from GL nodes.
    Params 8-15: log-weights.
    """
    gl_nodes, gl_weights = _gauss_legendre_init()
    # Node shifts: nodes_i = gl_nodes_i * exp(params[i])
    # This keeps nodes in a reasonable range near GL nodes
    node_shifts = params[:8]
    raw_nodes = gl_nodes * np.exp(node_shifts)
    # Clip to (0,1) and ensure strictly increasing
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing by small perturbation if needed
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    
    # Weights
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    
    return raw_nodes, weights

def _objective(params):
    if _time_left() < 2.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Test set: representative cosmologies and redshifts
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    return _worst_rel_error(nodes, weights, test_cosmos, test_zs)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Start from Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    
    # Initial params: zero shifts, log of GL weights
    params_init = np.zeros(16)
    params_init[8:16] = np.log(gl_weights)
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Try multiple optimization methods with restarts
    methods = ['Nelder-Mead', 'L-BFGS-B']
    
    for method in methods:
        if _time_left() < 10.0:
            break
            
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective, params_init, method=method, 
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            else:
                res = minimize(_objective, params_init, method=method, 
                               options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Refine from best found
    if _time_left() > 15.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final decode
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
