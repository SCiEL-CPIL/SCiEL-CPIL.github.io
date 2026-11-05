"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _E_squared(a, Om, w0, wa):
    """Vectorized E^2(a) for CPL model."""
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0 * w0)) * np.exp(-3.0 * wa * (1.0 - a))
    return np.maximum(term1 + term2, 1e-30)

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.dot(weights, 1.0 / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=500):
    """High-accuracy reference using vectorized Simpson's rule."""
    if z < 1e-12:
        return 0.0
    t_grid = np.linspace(0, 1, n_ref)
    a_vals = 1.0 / (1.0 + z * t_grid)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    f = 1.0 / E
    # Simpson's rule (n_ref must be odd for proper Simpson; use even with trapezoid fallback)
    if n_ref % 2 == 1:
        # Simpson's 1/3 rule
        h = 1.0 / (n_ref - 1)
        result = h / 3.0 * (f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-2:2]))
    else:
        from scipy.integrate import trapezoid
        result = trapezoid(f, t_grid)
    return (1.0 + z) * z * result

def _worst_case_error(nodes, weights, portfolio, z_values):
    """Compute worst relative error across portfolio, vectorized over z."""
    max_err = 0.0
    for Om, w0, wa in portfolio:
        # Vectorize over z
        z_arr = z_values
        # Quadrature values for all z at once
        # a_vals shape: (n_z, 8)
        a_mat = 1.0 / (1.0 + np.outer(z_arr, nodes))  # (n_z, 8)
        E2_mat = _E_squared(a_mat, Om, w0, wa)
        E_mat = np.sqrt(E2_mat)
        quad_vals = (1.0 + z_arr) * z_arr * np.sum(weights[None, :] / E_mat, axis=1)
        
        # Reference values
        ref_vals = np.array([_reference_value_fast(z, Om, w0, wa) for z in z_arr])
        
        mask = np.abs(ref_vals) > 1e-10
        if np.any(mask):
            rel_errs = np.abs(quad_vals[mask] - ref_vals[mask]) / np.abs(ref_vals[mask]) * 100.0
            if len(rel_errs) > 0:
                err = np.max(rel_errs)
                if err > max_err:
                    max_err = err
    return max_err

def _decode_params(x):
    """Decode parameters to valid nodes and weights.
    x[0:8] = logit parameters for nodes (mapped to (0,1) via sigmoid)
    x[8:16] = log parameters for weights (mapped via softmax)
    """
    # Nodes: use sigmoid to map to (0,1), then sort
    node_logits = x[:8]
    nodes = 1.0 / (1.0 + np.exp(-node_logits))
    # Shift to ensure they're well inside (0,1)
    nodes = 0.01 + 0.98 * nodes
    nodes = np.sort(nodes)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-8:
            nodes[i] = nodes[i-1] + 1e-8
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    
    # Weights: softmax
    w_logits = x[8:16]
    w_logits -= np.max(w_logits)
    w_exp = np.exp(w_logits)
    weights = w_exp / np.sum(w_exp)
    
    return nodes, weights

def construct():
    t_start = time.time()
    deadline = 85.0  # seconds, leave 5s margin for safety
    
    # Portfolio of cosmologies to optimize for - diverse set
    portfolio = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.5),   # CPL with large positive wa
        (0.30, -1.0, -0.5),  # CPL with large negative wa
        (0.25, -1.0, 0.0),   # Lower matter density
        (0.35, -1.0, 0.0),   # Higher matter density
        (0.30, -1.5, 0.5),   # Very negative w0
        (0.30, -0.5, -0.5),  # Less negative w0
        (0.28, -1.2, 0.3),   # Mixed
        (0.32, -0.8, -0.3),  # Mixed
        (0.30, -2.0, 1.0),   # Extreme
        (0.30, -0.5, 0.5),   # Extreme
        (0.20, -1.0, 0.0),   # Very low matter
    ]
    
    # z values to test - focus on where errors are largest
    z_values = np.concatenate([
        np.linspace(0.1, 1.0, 20),
        np.linspace(1.0, 2.0, 15),
        np.linspace(2.0, 3.0, 10),
    ])
    
    # Start from Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode starting point
    node_logits = np.log(nodes_gl / (1.0 - nodes_gl))
    w_logits = np.log(weights_gl + 1e-15)
    x0 = np.concatenate([node_logits, w_logits])
    
    def _objective(x):
        if time.time() - t_start > deadline:
            return 1e10
        nodes, weights = _decode_params(x)
        return _worst_case_error(nodes, weights, portfolio, z_values)
    
    # Initial evaluation
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Multiple optimization attempts
    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
    
    for attempt in range(20):
        if time.time() - t_start > deadline - 10.0:
            break
        
        # Perturb starting point for diversity
        if attempt == 0:
            x_start = x0.copy()
        else:
            x_start = best_x + np.random.normal(0, 0.1, size=16)
        
        method = methods[attempt % len(methods)]
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective, x_start, method='L-BFGS-B',
                             options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            else:  # Powell
                res = minimize(_objective, x_start, method='Powell',
                             options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final refinement with L-BFGS-B from best point
    if time.time() - t_start < deadline - 5.0:
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B',
                         options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    nodes, weights = _decode_params(best_x)
    
    # Final constraint enforcement
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    # Verify sum of weights is 1
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
