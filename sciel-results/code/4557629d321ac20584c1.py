"""Structural change: fully vectorized exact reference + smooth max objective + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    z = np.asarray(z, dtype=np.float64)
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Precompute high-order Gauss-Legendre for exact reference
_N_EXACT = 256
_XS_EXACT, _WS_EXACT = np.polynomial.legendre.leggauss(_N_EXACT)


def _exact_comoving_vec(z_arr, w0, wa):
    """Vectorized exact comoving distance for array of z values.
    For each z, computes ∫₀ᶻ dz'/E(z') via Gauss-Legendre on [0, z].
    Returns array same shape as z_arr.
    """
    z_arr = np.asarray(z_arr, dtype=np.float64)
    n = len(z_arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        zi = z_arr[i]
        if zi <= 0:
            result[i] = 0.0
            continue
        z_pts = 0.5 * zi * (_XS_EXACT + 1.0)
        E_pts = _cpl_E(z_pts, w0, wa)
        result[i] = 0.5 * zi * np.sum(_WS_EXACT / E_pts)
    return result


def _quad_comoving_vec(z_arr, w0, wa, nodes, weights):
    """Vectorized quadrature comoving distance for array of z values.
    quad(z) = z * sum_i weights[i] / E(z * nodes[i])
    """
    z_arr = np.asarray(z_arr, dtype=np.float64)
    n_z = len(z_arr)
    # z_nodes shape: (n_z, 8)
    z_nodes = np.outer(z_arr, nodes)
    E_mat = _cpl_E(z_nodes, w0, wa)  # (n_z, 8)
    return z_arr * np.sum(weights[None, :] / E_mat, axis=1)


def _compute_all_rel_errors(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute all relative errors, return flat array."""
    z_arr = np.asarray(z_vals, dtype=np.float64)
    all_errors = []
    for w0 in w0_vals:
        for wa in wa_vals:
            exact = _exact_comoving_vec(z_arr, w0, wa)
            quad = _quad_comoving_vec(z_arr, w0, wa, nodes, weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.abs(quad - exact) / np.maximum(np.abs(exact), 1e-30)
            all_errors.append(rel)
    return np.concatenate(all_errors)


def _smooth_max(errors, temperature=0.005):
    """Smooth approximation of max via log-sum-exp."""
    errors = np.asarray(errors, dtype=np.float64)
    m = np.max(errors)
    if m < 1e-30:
        return m
    # log-sum-exp scaled
    scaled = errors / temperature
    sm = np.max(scaled)
    val = m + temperature * (sm + np.log(np.sum(np.exp(scaled - sm))))
    return val


def _worst_error(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute worst relative error over ensemble."""
    errs = _compute_all_rel_errors(nodes, weights, w0_vals, wa_vals, z_vals)
    return float(np.max(errs))


def _objective_smooth(params, nodes_raw, weights_raw, w0_vals, wa_vals, z_vals, temp):
    """Smooth objective using log-sum-exp of relative errors."""
    if not _deadline_ok():
        return 1e10
    nodes_p, weights_p = _params_to_nw(params)
    errs = _compute_all_rel_errors(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
    return _smooth_max(errs, temp)


def _params_to_nw(params):
    """Convert unconstrained params to valid (nodes, weights)."""
    n = 8
    x = params[:n]
    y = params[n:]
    # Nodes: softplus-like via cumulative logit
    # Use sigmoid to map to (0,1) and cumsum for ordering
    sig = 1.0 / (1.0 + np.exp(-x))
    # Ensure strictly increasing: add small monotone offset
    offsets = np.arange(1, n + 1, dtype=np.float64) * 0.1
    raw = sig * 0.8 + offsets * 0.2
    nodes_p = raw / raw[-1]
    eps = 1e-10
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i-1]:
            nodes_p[i] = nodes_p[i-1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    # Weights: softmax
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    return nodes_p, weights_p


def _nw_to_params(nodes, weights):
    """Invert _params_to_nw approximately for initialization."""
    n = 8
    # For nodes: we used sig * 0.8 + offsets * 0.2, then normalized
    # Just use log for a reasonable starting point
    x = np.log(nodes)
    y = np.log(weights)
    return np.concatenate([x, y])


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize using differential evolution + local polish."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    try:
        from scipy.optimize import differential_evolution, minimize
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if not have_scipy:
        return best_nodes, best_weights
    
    # Define bounds for DE
    # x (nodes): use range that maps to reasonable node positions
    # y (weights): use range that maps to reasonable weights
    bounds = [(-5.0, 5.0)] * n + [(-10.0, 10.0)] * n
    
    # Smooth objective for DE
    temp = 0.01
    def obj_de(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        errs = _compute_all_rel_errors(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
        return _smooth_max(errs, temp)
    
    # Initial point
    p0 = _nw_to_params(best_nodes, best_weights)
    
    t_start = time.time()
    
    # Phase 1: Differential Evolution (global search)
    try:
        res_de = differential_evolution(
            obj_de, bounds,
            seed=42,
            maxiter=80,
            popsize=12,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            init='sobol',
            polish=False,
            updating='immediate'
        )
        if _deadline_ok():
            np_de, wp_de = _params_to_nw(res_de.x)
            err_de = _worst_error(np_de, wp_de, w0_vals, wa_vals, z_vals)
            if err_de < best_err:
                best_nodes, best_weights, best_err = np_de, wp_de, err_de
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead local polish from best
    if _deadline_ok():
        p_cur = _nw_to_params(best_nodes, best_weights)
        
        def obj_nm(params):
            if not _deadline_ok():
                return 1e10
            nodes_p, weights_p = _params_to_nw(params)
            errs = _compute_all_rel_errors(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
            return _smooth_max(errs, 0.003)
        
        try:
            res_nm = minimize(
                obj_nm, p_cur,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                         'maxfev': 400}
            )
            if _deadline_ok():
                np_nm, wp_nm = _params_to_nw(res_nm.x)
                err_nm = _worst_error(np_nm, wp_nm, w0_vals, wa_vals, z_vals)
                if err_nm < best_err:
                    best_nodes, best_weights, best_err = np_nm, wp_nm, err_nm
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if _deadline_ok():
        for iteration in range(10):
            if not _deadline_ok():
                break
            improved = False
            # Perturb each node
            for i in range(n):
                if not _deadline_ok():
                    break
                for delta in [1e-5, -1e-5, 5e-5, -5e-5, 1e-4, -1e-4, 2e-5, -2e-5]:
                    new_nodes = best_nodes.copy()
                    new_nodes[i] += delta
                    valid = all(0 < new_nodes[j] < 1 for j in range(n))
                    if valid and all(new_nodes[j] > new_nodes[j-1] for j in range(1, n)):
                        err_new = _worst_error(new_nodes, best_weights, w0_vals, wa_vals, z_vals)
                        if err_new < best_err:
                            best_nodes = new_nodes
                            best_err = err_new
                            improved = True
            # Perturb each weight
            for i in range(n):
                if not _deadline_ok():
                    break
                for delta in [1e-5, -1e-5, 5e-5, -5e-5, 1e-4, -1e-4]:
                    new_weights = best_weights.copy()
                    new_weights[i] += delta
                    if np.all(new_weights > 0):
                        new_weights = new_weights / new_weights.sum()
                        err_new = _worst_error(best_nodes, new_weights, w0_vals, wa_vals, z_vals)
                        if err_new < best_err:
                            best_weights = new_weights
                            best_err = err_new
                            improved = True
            if not improved:
                break
    
    best_weights = best_weights / best_weights.sum()
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    nodes_init = np.clip(nodes_init, 1e-15, 1.0 - 1e-15)
    for i in range(1, n):
        if nodes_init[i] <= nodes_init[i-1]:
            nodes_init[i] = nodes_init[i-1] + 1e-15
    nodes_init[-1] = min(nodes_init[-1], 1.0 - 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    # Broader ensemble for optimization
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5, 0.8]
    wa_vals = [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.2]
    z_vals = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    
    time_budget = 80.0
    
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
