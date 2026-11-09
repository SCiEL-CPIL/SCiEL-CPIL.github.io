"""Direct 8-node quadrature construction using global search and pre-computed references."""
import time
import numpy as np
from scipy.optimize import differential_evolution

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

def _params_to_nodes_weights(params):
    """
    Direct parameterization: params 0-7 are node positions via sigmoid,
    params 8-15 are log-weights.
    """
    raw_nodes = 1.0 / (1.0 + np.exp(-params[:8]))
    raw_nodes = np.sort(raw_nodes)
    # Ensure strict increase
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    return raw_nodes, weights

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Pre-compute reference data
    # Test set
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (1.0, -2.0), (-2.0, 1.0),
        (0.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (-0.5, -0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # Reference nodes (64-pt GL)
    ref_x, ref_w = np.polynomial.legendre.leggauss(64)
    ref_nodes = 0.5 * (ref_x + 1.0)
    ref_weights = 0.5 * ref_w
    
    # Pre-compute true dL for all combinations
    # Shape: (n_cosmos, n_zs)
    true_dLs = np.zeros((len(test_cosmos), len(test_zs)))
    for i, (w0, wa) in enumerate(test_cosmos):
        for j, z in enumerate(test_zs):
            a_ref = 1.0 / (1.0 + z * ref_nodes)
            E_ref = _E_cpl(a_ref, w0, wa)
            ref_integral = np.sum(ref_weights / E_ref)
            true_dLs[i, j] = (1.0 + z) * z * ref_integral
            
    # Store z values for error calculation
    z_grid = np.tile(test_zs, (len(test_cosmos), 1))
    
    # Initial guess: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    params_init = np.zeros(16)
    # Inverse sigmoid
    params_init[:8] = np.log(nodes_gl / (1 - nodes_gl))
    params_init[8:16] = np.log(weights_gl)
    
    best_params = params_init.copy()
    
    def objective(params):
        if _time_left() < 5.0:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        
        # Vectorized calculation of approx dLs
        # nodes: (8,), weights: (8,)
        # We need to compute for each (cosmo, z)
        # a_8 = 1 / (1 + z * nodes)
        # E_8 = _E_cpl(a_8, w0, wa)
        # approx_int = sum(weights / E_8)
        # approx_dL = (1+z)*z*approx_int
        
        max_err = 0.0
        for i, (w0, wa) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                a_8 = 1.0 / (1.0 + z * nodes)
                E_8 = _E_cpl(a_8, w0, wa)
                approx_integral = np.sum(weights / E_8)
                approx_dL = (1.0 + z) * z * approx_integral
                true_dL = true_dLs[i, j]
                if true_dL > 1e-10:
                    rel_err = abs(approx_dL - true_dL) / true_dL
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err
    
    # Get initial score
    best_score = objective(best_params)
    
    # Global Optimization: Differential Evolution
    # Bounds for sigmoid params: [-10, 10] is enough to cover (0,1)
    # Bounds for log weights: [-5, 5]
    bounds = [(-10, 10)] * 8 + [(-5, 5)] * 8
    
    if _time_left() > 10.0:
        try:
            # DE is stochastic, set seed for reproducibility if needed, but let it run
            # maxiter and popsize control the budget
            # 16 dimensions. popsize=15 -> 240 evals per iter.
            # 100 iters -> 24000 evals.
            # Each eval is ~120 * (8 evals of E_cpl + sums). ~1000 flops.
            # 24000 * 1000 = 24M flops. Should be fast (<1s).
            # We can afford more iterations.
            res = differential_evolution(objective, bounds, maxiter=200, popsize=15, 
                                         tol=1e-8, seed=42, polish=True)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Local Polish with Nelder-Mead
    if _time_left() > 5.0:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    nodes, weights = _params_to_nodes_weights(best_params)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    # Final sanity check
    if not np.all(nodes < 1.0) or not np.all(nodes > 0.0):
        # Fallback to GL if something went wrong
        nodes, weights = nodes_gl, weights_gl
        
    return {"nodes": nodes, "weights": weights}
