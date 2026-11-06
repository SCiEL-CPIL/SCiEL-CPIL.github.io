"""Improved program: Fast vectorized optimization of 8-node universal quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    # Handle z=0 carefully to avoid 0**0 or similar issues, though z>0 in integration
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    # E(z)^2 = (1+z)^(3(1+w0+wa)) * (1+wa*z)^(-3*wa)
    # For stability, use logs if possible, but direct power is usually fine for z in [0, 5]
    term1 = np.power(zp1, exp1)
    term2 = np.power(1.0 + wa * z, -3.0 * wa)
    return np.sqrt(term1 * term2)


def _exact_comoving_vectorized(z_arr, w0, wa, n_quad=100):
    """
    Exact comoving distances for array of z values via Gauss-Legendre.
    Vectorized over z for speed.
    """
    # Pre-compute GL nodes/weights once outside if called repeatedly, but here it's simple enough
    # To maximize speed in the loop, we assume this is called many times.
    # Better: Pre-compute GL nodes and weights globally or pass them.
    # For now, compute inside but use a fixed small n_quad for speed.
    # Actually, let's pre-compute global GL nodes for the "exact" reference.
    # But the function signature doesn't allow passing them easily without changing interface.
    # We can store them in a global variable or compute once.
    # Let's compute once and cache.
    if not hasattr(_exact_comoving_vectorized, '_gl_cache'):
        _exact_comoving_vectorized._gl_cache = {}
    if n_quad not in _exact_comoving_vectorized._gl_cache:
        xs, ws = np.polynomial.legendre.leggauss(n_quad)
        _exact_comoving_vectorized._gl_cache[n_quad] = (xs, ws)
    else:
        xs, ws = _exact_comoving_vectorized._gl_cache[n_quad]
    
    n_z = len(z_arr)
    exact_arr = np.zeros(n_z)
    
    # Vectorize over z and GL nodes
    # z_vals shape: (n_z, n_quad)
    # u in [0,1], z' = z * u
    # GL nodes xs are in [-1, 1], map to [0, 1] via u = 0.5*(xs+1)
    u = 0.5 * (xs + 1.0) # shape (n_quad,)
    w_u = 0.5 * ws       # shape (n_quad,)
    
    # z_arr shape (n_z,), u shape (n_quad,)
    # z_vals = z_arr[:, None] * u[None, :] -> shape (n_z, n_quad)
    z_vals = z_arr[:, None] * u[None, :]
    
    # E_vals shape (n_z, n_quad)
    E_vals = _cpl_E(z_vals, w0, wa)
    
    # Integrand: 1/E(z')
    # Integral: sum_i w_u[i] * (1/E(z_i)) * z
    # exact_arr[j] = z[j] * sum_i w_u[i] / E(z[j] * u[i])
    
    inv_E = 1.0 / E_vals
    # Sum over axis 1 (GL nodes)
    integral_sum = np.sum(w_u[None, :] * inv_E, axis=1)
    exact_arr = z_arr * integral_sum
    
    return exact_arr


def _quad_comoving_vectorized(z_arr, w0, wa, nodes, weights):
    """
    Quadrature comoving distances for array of z values.
    Vectorized.
    """
    n_z = len(z_arr)
    n_nodes = len(nodes)
    
    # z_vals shape: (n_z, n_nodes)
    z_vals = z_arr[:, None] * nodes[None, :]
    
    # E_vals shape: (n_z, n_nodes)
    E_vals = _cpl_E(z_vals, w0, wa)
    
    # Integrand: 1/E(z')
    inv_E = 1.0 / E_vals
    
    # Sum over axis 1 (nodes)
    # quad_arr[j] = z[j] * sum_i weights[i] / E(z[j] * nodes[i])
    integral_sum = np.sum(weights[None, :] * inv_E, axis=1)
    quad_arr = z_arr * integral_sum
    
    return quad_arr


def _worst_error(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=100):
    """Compute worst relative error over ensemble."""
    worst = 0.0
    z_arr = np.array(z_vals)
    
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_vectorized(z_arr, w0, wa, n_quad)
            quad_arr = _quad_comoving_vectorized(z_arr, w0, wa, nodes, weights)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            
            current_worst = float(np.max(rel_err))
            if current_worst > worst:
                worst = current_worst
                
    return worst


def _params_to_nw(params):
    """Convert parameters to nodes and weights, ensuring constraints."""
    n = 8
    x = params[:n]
    y = params[n:]
    
    # Nodes: strictly increasing in (0, 1)
    # Use cumsum of exp(x) to ensure positivity and ordering
    ex = np.exp(x - np.max(x))
    cumsum = np.cumsum(ex)
    nodes_p = cumsum / cumsum[-1]
    
    # Ensure strictly increasing and within (0, 1)
    eps = 1e-12
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i-1]:
            nodes_p[i] = nodes_p[i-1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    
    # Weights: positive and sum to 1
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    
    return nodes_p, weights_p


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """
    Optimize nodes and weights to minimize worst-case error.
    """
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
    
    # Initial parameters
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    y_init = np.log(np.maximum(best_weights, 1e-15))
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    
    # 1. Nelder-Mead for global-ish search
    res = minimize(
        _objective, p0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                 'maxfev': 500}
    )
    
    if _deadline_ok():
        nodes_nm, weights_nm = _params_to_nw(res.x)
        err_nm = _worst_error(nodes_nm, weights_nm, w0_vals, wa_vals, z_vals)
        if err_nm < best_err:
            best_nodes, best_weights, best_err = nodes_nm, weights_nm, err_nm
    
    # 2. L-BFGS-B for local refinement
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.5:
        # Start from the best found so far
        p_cur = res.x if res.success else p0
        # Refine p_cur if it's better
        nodes_cur, weights_cur = _params_to_nw(p_cur)
        err_cur = _worst_error(nodes_cur, weights_cur, w0_vals, wa_vals, z_vals)
        if err_cur < best_err:
            best_nodes, best_weights, best_err = nodes_cur, weights_cur, err_cur
            # Re-initialize p_cur from best
            x_best = np.log(np.maximum(best_nodes, 1e-15))
            y_best = np.log(np.maximum(best_weights, 1e-15))
            p_cur = np.concatenate([x_best, y_best])
            
        res2 = minimize(
            _objective, p_cur,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-12, 'maxfun': 300}
        )
        
        if _deadline_ok():
            nodes_lb, weights_lb = _params_to_nw(res2.x)
            err_lb = _worst_error(nodes_lb, weights_lb, w0_vals, wa_vals, z_vals)
            if err_lb < best_err:
                best_nodes, best_weights, best_err = nodes_lb, weights_lb, err_lb
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Initialize with Gauss-Legendre nodes and weights on [0, 1]
    # This is a strong starting point for smooth integrands
    xs, ws = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [0, 1]
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    
    # Ensure strict ordering and bounds (GL nodes are already distinct and in [-1,1])
    nodes_init = np.sort(nodes_init)
    nodes_init = np.clip(nodes_init, 1e-12, 1.0 - 1e-12)
    # Weights are already positive and sum to 1 (approx)
    weights_init = weights_init / weights_init.sum()
    
    # Ensemble for optimization
    # Diverse set of w0, wa
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    
    # Redshifts: cover a wide range
    z_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    time_budget = 78.0 # Leave 2s margin
    
    try:
        nodes, weights = _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
    except Exception:
        nodes, weights = nodes_init, weights_init
    
    # Final cleanup
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    nodes[-1] = min(nodes[-1], 1.0 - 1e-12)
    weights = np.maximum(weights, 1e-15)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
