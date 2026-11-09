"""Gauss-Legendre-based 8-node quadrature for luminosity distance with hierarchical optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog
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

def _true_dL_fast(z, w0, wa, n_ref=200):
    x_ref, w_ref = np.polynomial.legendre.leggauss(n_ref)
    x_ref = 0.5 * (x_ref + 1.0)
    w_ref = 0.5 * w_ref
    a_vals = 1.0 / (1.0 + z * x_ref)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(w_ref / E_vals)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _solve_weights_lp(nodes, test_cosmos, test_zs):
    """Solve LP to find weights minimizing max relative error."""
    # Build constraint matrix: for each (cosmo, z) pair, we want
    # |quad_val - true_val| / true_val <= eps
    # quad_val = (1+z)*z * sum(w_i / E_i)
    # Let c_i = (1+z)*z / E_i, so quad_val = sum(c_i * w_i)
    # Relative error: |sum(c_i * w_i) - true_val| / true_val <= eps
    # This is: -eps*true_val <= sum(c_i * w_i) - true_val <= eps*true_val
    
    n_weights = 8
    n_constraints = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            n_constraints += 2  # upper and lower bound
    
    # A_ub * w <= b_ub
    A_ub = np.zeros((n_constraints, n_weights))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            c_vals = (1.0 + z) * z / E_vals
            true_val = _true_dL_fast(z, w0, wa)
            eps = 1e-10  # We'll scale this
            
            # Upper: sum(c_i * w_i) - true_val <= eps * true_val
            A_ub[row, :] = c_vals
            b_ub[row] = true_val + 1e-10 * true_val
            row += 1
            
            # Lower: -(sum(c_i * w_i) - true_val) <= eps * true_val
            A_ub[row, :] = -c_vals
            b_ub[row] = -true_val + 1e-10 * true_val
            row += 1
    
    # Equality constraint: sum(w_i) = 1
    A_eq = np.ones((1, n_weights))
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0
    bounds = [(0, None)] * n_weights
    
    try:
        res = linprog(c=np.zeros(n_weights), A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x
    except Exception:
        pass
    
    # Fallback: equal weights
    return np.ones(n_weights) / n_weights

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _params_to_nodes(params):
    """Convert 8 params to strictly increasing nodes in (0,1)."""
    # Use sigmoid-like mapping to ensure nodes in (0,1)
    nodes = 1.0 / (1.0 + np.exp(-params))
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    # Clip to safe range
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    return nodes

def _objective_nodes(params, test_cosmos, test_zs):
    if _time_left() < 2.0:
        return 1e10
    
    nodes = _params_to_nodes(params)
    weights = _solve_weights_lp(nodes, test_cosmos, test_zs)
    return _worst_rel_error(nodes, weights, test_cosmos, test_zs)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Test set
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (0.0, -0.5),   # CPL variation
        (-0.8, 0.3),   # Another CPL
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    # Initialize from Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    # Invert sigmoid to get initial params
    params_init = np.log(gl_nodes / (1.0 - gl_nodes))
    
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    best_score = _worst_rel_error(best_nodes, best_weights, test_cosmos, test_zs)
    
    # Optimize nodes using Nelder-Mead, with LP for weights at each step
    if _time_left() > 20.0:
        try:
            res = minimize(_objective_nodes, params_init, 
                          args=(test_cosmos, test_zs),
                          method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_nodes = _params_to_nodes(best_params)
                best_weights = _solve_weights_lp(best_nodes, test_cosmos, test_zs)
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if _time_left() < 15.0:
            break
        try:
            params_rand = np.random.randn(8) * 0.5
            res = minimize(_objective_nodes, params_rand,
                          args=(test_cosmos, test_zs),
                          method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_nodes = _params_to_nodes(best_params)
                best_weights = _solve_weights_lp(best_nodes, test_cosmos, test_zs)
        except Exception:
            pass
    
    # Final refinement
    if _time_left() > 10.0:
        try:
            res = minimize(_objective_nodes, best_params,
                          args=(test_cosmos, test_zs),
                          method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_nodes = _params_to_nodes(res.x)
                best_weights = _solve_weights_lp(best_nodes, test_cosmos, test_zs)
        except Exception:
            pass
    
    # Ensure validity
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
