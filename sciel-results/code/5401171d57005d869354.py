"""Model-and-optimize approach with moment-matching initialization and refined search."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 88.0

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

# Precompute high-accuracy Gauss-Legendre nodes/weights for reference integration
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    if z == 0:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

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

def _softmax_error(nodes, weights, test_cosmos, test_zs, beta=100.0):
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    if not errors:
        return 0.0
    errors = np.array(errors)
    max_e = np.max(errors)
    if max_e < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (errors - max_e)))) / beta + max_e
    return log_sum

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _moment_match_init(w0, wa, z_ref=1.0):
    """
    Initialize nodes/weights by matching moments of 1/E(a) for a specific cosmology.
    This creates a rule that is exact for the specific shape of 1/E(a) in that model.
    """
    # We want sum_i w_i * f(x_i) ≈ integral_0^1 f(x) dx for f(x) = 1/E(1/(1+z_ref*x))
    # Let's pick nodes as Gauss-Legendre nodes, then adjust weights to match
    # the integral of f, f^2, etc. But with 8 points, we can match 8 moments.
    # Actually, simpler: use GL nodes, but re-weight to match the integral exactly.
    # Even better: use a least-squares fit to match the integral over a range of z.
    
    # For now, let's just use GL nodes but compute optimal weights for a specific z
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    
    # Compute 1/E(a) at the nodes for the reference cosmology
    a_vals = 1.0 / (1.0 + z_ref * nodes)
    f_vals = 1.0 / _E_cpl(a_vals, w0, wa)
    
    # The true integral of f from 0 to z_ref is:
    true_integral = _true_dL(z_ref, w0, wa) / (1.0 + z_ref)
    
    # We want sum_i w_i * f_i ≈ true_integral / z_ref (since dL = (1+z)*z*sum)
    # Actually: dL/(c/H0) = (1+z) * z * sum_i w_i/E(a_i)
    # So sum_i w_i/E(a_i) ≈ integral_0^z 1/E dz' / z
    # Let's just normalize GL weights so that the quadrature matches the true integral
    integral_approx = np.sum(w_gl * f_vals)
    # Scale weights to match
    if integral_approx > 1e-15:
        scale = true_integral / (z_ref * integral_approx)
        weights = w_gl * scale
    else:
        weights = w_gl.copy()
    
    weights = weights / np.sum(weights)
    return nodes, weights

def _decode_params(params):
    raw = params[:8]
    sig = 1.0 / (1.0 + np.exp(-10.0 * (raw - 3.5)))
    sig = np.sort(sig)
    for i in range(1, 8):
        if sig[i] <= sig[i-1] + 1e-12:
            sig[i] = sig[i-1] + 1e-12
    sig = np.clip(sig, 1e-10, 1.0 - 1e-10)
    nodes = sig
    
    log_w = params[8:16]
    w_raw = np.exp(log_w)
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _encode_initial(nodes, weights):
    params_nodes = np.log(nodes / (1.0 - nodes)) / 10.0 + 3.5
    params_w = np.log(weights)
    return np.concatenate([params_nodes, params_w])

def _objective(params):
    if _time_left() < 3.0:
        return 1e10
    
    nodes, weights = _decode_params(params)
    
    test_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (-2.0, 1.0),
        (1.0, -2.0),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    return _softmax_error(nodes, weights, test_cosmos, test_zs, beta=80.0)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 88.0
    
    np.random.seed(42)
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Standard Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    initializations.append(_encode_initial(gl_nodes, gl_weights))
    
    # 2. Moment-matched initializations for several representative cosmologies
    mm_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (0.0, 0.0),  # LCDM again with different z_ref
    ]
    for w0, wa in mm_cosmos:
        for z_ref in [0.5, 1.0, 2.0]:
            mm_nodes, mm_weights = _moment_match_init(w0, wa, z_ref)
            initializations.append(_encode_initial(mm_nodes, mm_weights))
    
    # 3. Random perturbations around GL
    for _ in range(15):
        perturbation = np.random.randn(16) * 0.5
        initializations.append(initializations[0] + perturbation)
    
    best_params = initializations[0].copy()
    best_score = _objective(best_params)
    
    # Optimize from each initialization
    for init_params in initializations:
        if _time_left() < 5.0:
            break
        
        try:
            res = minimize(_objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'initial_simplex': 0.1})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if _time_left() > 5.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Decode final parameters
    nodes, weights = _decode_params(best_params)
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    weights = np.abs(weights)
    if np.sum(weights) < 1e-15:
        weights = np.ones(8) / 8.0
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
