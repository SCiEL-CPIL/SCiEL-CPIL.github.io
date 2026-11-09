"""8-node positive quadrature for luminosity distance — fast smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize

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
    return (1.0 + z) * z * integral

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

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
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _params_to_nodes_weights(params):
    gl_nodes, gl_weights = _gauss_legendre_init()
    node_shifts = params[:8]
    raw_nodes = gl_nodes * np.exp(node_shifts)
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    return raw_nodes, weights

def _smooth_objective(params, temperature=0.001):
    if _time_left() < 2.0:
        return 1e10
    nodes, weights = _params_to_nodes_weights(params)
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0),
        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3),
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    errors = np.array(errors)
    if len(errors) == 0:
        return 1e10
    # Smooth max via log-sum-exp
    max_err = np.max(errors)
    log_sum = np.log(np.sum(np.exp((errors - max_err) / temperature))) + max_err / temperature
    return np.exp(log_sum) * temperature

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    gl_nodes, gl_weights = _gauss_legendre_init()
    params_init = np.zeros(16)
    params_init[8:16] = np.log(gl_weights)
    
    best_params = params_init.copy()
    best_score = _worst_rel_error(*_params_to_nodes_weights(best_params), 
                                  [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                   (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                  np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
    
    # Phase 1: L-BFGS-B with smooth objective
    if _time_left() > 15.0:
        try:
            res = minimize(_smooth_objective, params_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
            cand_nodes, cand_weights = _params_to_nodes_weights(res.x)
            cand_err = _worst_rel_error(cand_nodes, cand_weights,
                                       [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                       np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
            if cand_err < best_score:
                best_score = cand_err
                best_params = res.x
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement with true max error
    if _time_left() > 10.0:
        try:
            def _true_obj(params):
                if _time_left() < 2.0:
                    return 1e10
                nodes, weights = _params_to_nodes_weights(params)
                return _worst_rel_error(nodes, weights,
                                       [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                       np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
            
            res = minimize(_true_obj, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            cand_nodes, cand_weights = _params_to_nodes_weights(res.x)
            cand_err = _worst_rel_error(cand_nodes, cand_weights,
                                       [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                       np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
            if cand_err < best_score:
                best_score = cand_err
                best_params = res.x
        except Exception:
            pass
    
    # Phase 3: Final fine-tune
    if _time_left() > 5.0:
        try:
            def _true_obj2(params):
                if _time_left() < 1.0:
                    return 1e10
                nodes, weights = _params_to_nodes_weights(params)
                return _worst_rel_error(nodes, weights,
                                       [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                       np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
            
            res = minimize(_true_obj2, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
            cand_nodes, cand_weights = _params_to_nodes_weights(res.x)
            cand_err = _worst_rel_error(cand_nodes, cand_weights,
                                       [(0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), 
                                        (-1.5, 0.5), (-0.5, 0.5), (0.3, -0.3)],
                                       np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]))
            if cand_err < best_score:
                best_score = cand_err
                best_params = res.x
        except Exception:
            pass
    
    nodes, weights = _params_to_nodes_weights(best_params)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
