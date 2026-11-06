"""Improved program: fast vectorized alternating optimization of 8-node universal quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving_vectorized(z_arr, w0, wa, n_quad=200):
    """Exact comoving distances for array of z values via Gauss-Legendre (vectorized over z)."""
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    # z_arr shape (nz,), xs shape (nq,)
    # z_vals shape (nz, nq)
    z_vals = 0.5 * np.outer(z_arr, np.ones(n_quad)) * (xs + 1.0)  # (nz, nq)
    E_vals = _cpl_E(z_vals, w0, wa)  # (nz, nq)
    exact_arr = 0.5 * z_arr * np.sum(ws / E_vals, axis=1)
    return exact_arr


def _quad_comoving_vectorized(z_arr, w0, wa, nodes, weights):
    """Quadrature comoving distances for array of z values (vectorized)."""
    # z_arr shape (nz,), nodes shape (8,)
    z_vals = np.outer(z_arr, nodes)  # (nz, 8)
    E_vals = _cpl_E(z_vals, w0, wa)  # (nz, 8)
    quad_arr = z_arr * np.sum(weights / E_vals, axis=1)
    return quad_arr


def _worst_error_fast(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=200):
    """Compute worst relative error over ensemble (vectorized)."""
    worst = 0.0
    z_arr = np.array(z_vals)
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_vectorized(z_arr, w0, wa, n_quad)
            quad_arr = _quad_comoving_vectorized(z_arr, w0, wa, nodes, weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            worst = max(worst, float(np.max(rel_err)))
    return worst


def _optimize_weights_given_nodes(nodes, w0_vals, wa_vals, z_vals, n_quad=200):
    """
    Given fixed nodes, find optimal weights by solving a linear system.
    We want to minimize worst-case relative error, but as a proxy we match
    the integral for each (w0, wa, z) combination. Since weights appear linearly
    in the quadrature sum, we can set up a least-squares problem.
    
    For each cosmology (w0,wa) and redshift z, the quadrature gives:
      sum_i w_i / E(z * nodes_i)
    and the exact value is known. So we solve:
      A @ w = b  (least squares)
    where A[j, i] = 1/E(z_j * nodes_i) and b[j] = exact_j / z_j
    """
    z_arr = np.array(z_vals)
    n_z = len(z_arr)
    n = len(nodes)
    
    # Build system: for each (w0, wa, z), we have one equation
    A_rows = []
    b_vals = []
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_vectorized(z_arr, w0, wa, n_quad)
            for j, z in enumerate(z_arr):
                if z <= 0:
                    continue
                z_node = z * nodes
                E_vals = _cpl_E(z_node, w0, wa)
                A_rows.append(1.0 / E_vals)
                b_vals.append(exact_arr[j] / z)
    
    A = np.array(A_rows)  # (n_eq, n)
    b = np.array(b_vals)  # (n_eq,)
    
    # Solve least squares with non-negativity constraint
    # First try unconstrained
    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # If any weights are negative, use scipy's nnls or project
    if np.any(w_sol < 0):
        try:
            from scipy.optimize import nnls
            w_sol, _ = nnls(A, b)
        except ImportError:
            w_sol = np.maximum(w_sol, 0)
    
    w_sol = np.maximum(w_sol, 1e-15)
    w_sol = w_sol / w_sol.sum()
    return w_sol


def _optimize_nodes_given_weights(weights, w0_vals, wa_vals, z_vals, n_quad=200, max_iter=50):
    """
    Given fixed weights, optimize nodes using Nelder-Mead on just 8 variables.
    """
    from scipy.optimize import minimize
    
    n = 8
    best_nodes = np.linspace(0.1, 0.9, n)
    best_err = _worst_error_fast(best_nodes, weights, w0_vals, wa_vals, z_vals, n_quad)
    
    def _node_objective(params):
        if not _deadline_ok():
            return 1e10
        x = params
        ex = np.exp(x - np.max(x))
        cumsum = np.cumsum(ex)
        nodes_p = cumsum / cumsum[-1]
        eps = 1e-12
        nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
        for i in range(1, n):
            if nodes_p[i] <= nodes_p[i-1]:
                nodes_p[i] = nodes_p[i-1] + eps
        nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
        return _worst_error_fast(nodes_p, weights, w0_vals, wa_vals, z_vals, n_quad)
    
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    
    res = minimize(
        _node_objective, x_init,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                 'maxfev': max_iter * 2}
    )
    
    x_ = res.x
    ex = np.exp(x_ - np.max(x_))
    cumsum = np.cumsum(ex)
    nodes_opt = cumsum / cumsum[-1]
    nodes_opt = np.clip(nodes_opt, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if nodes_opt[i] <= nodes_opt[i-1]:
            nodes_opt[i] = nodes_opt[i-1] + 1e-12
    nodes_opt[-1] = min(nodes_opt[-1], 1.0 - 1e-12)
    
    err_opt = _worst_error_fast(nodes_opt, weights, w0_vals, wa_vals, z_vals, n_quad)
    if err_opt < best_err:
        best_nodes, best_err = nodes_opt, err_opt
    
    return best_nodes


def _alternating_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, n_quad=200, max_rounds=20):
    """
    Alternating optimization: fix nodes -> optimize weights, fix weights -> optimize nodes.
    """
    nodes = nodes_init.copy()
    weights = weights_init.copy()
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = _worst_error_fast(nodes, weights, w0_vals, wa_vals, z_vals, n_quad)
    
    for rnd in range(max_rounds):
        if not _deadline_ok():
            break
        
        # Step 1: Optimize weights given nodes
        weights = _optimize_weights_given_nodes(nodes, w0_vals, wa_vals, z_vals, n_quad)
        
        # Step 2: Optimize nodes given weights
        nodes = _optimize_nodes_given_weights(weights, w0_vals, wa_vals, z_vals, n_quad)
        
        # Evaluate
        err = _worst_error_fast(nodes, weights, w0_vals, wa_vals, z_vals, n_quad)
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    return best_nodes, best_weights


def construct():
    n = 8
    
    # Initialize with Gauss-Legendre nodes on [0,1] (optimal for smooth functions)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (gl_nodes + 1.0)
    nodes_init = np.sort(nodes_init)
    weights_init = gl_weights / gl_weights.sum()
    
    # Compact ensemble for fast optimization
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, 0.0, 0.5]
    z_vals = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Alternating optimization
    try:
        nodes, weights = _alternating_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals,
                                               n_quad=200, max_rounds=15)
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
