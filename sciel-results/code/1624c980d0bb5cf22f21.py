"""Improved program: smooth min-max optimization for 8-node quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving_cached(z, w0, wa):
    """Exact comoving distance via adaptive Gauss-Kronrod (cached)."""
    if z <= 0:
        return 0.0
    key = (round(z, 10), round(w0, 10), round(wa, 10))
    if key not in _exact_cache:
        from scipy.integrate import quad
        f = lambda zz: 1.0 / _cpl_E(zz, w0, wa)
        val, _ = quad(f, 0, z, limit=200, epsabs=1e-14, epsrel=1e-14)
        _exact_cache[key] = val
    return _exact_cache[key]


_exact_cache = {}


def _quad_comoving_vec(z_arr, nodes, weights, w0, wa):
    """Vectorized quadrature comoving distance for array of z values."""
    if len(z_arr) == 0:
        return np.array([])
    z_nodes = np.outer(z_arr, nodes)  # (n_z, 8)
    E_mat = _cpl_E(z_nodes, w0, wa)   # (n_z, 8)
    return z_arr * np.sum(weights[None, :] / E_mat, axis=1)


def _all_relative_errors(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute all relative errors as a flat array."""
    errors = []
    z_arr = np.array(z_vals)
    for w0 in w0_vals:
        for wa in wa_vals:
            quad_arr = _quad_comoving_vec(z_arr, nodes, weights, w0, wa)
            exact_arr = np.array([_exact_comoving_cached(z, w0, wa) for z in z_vals])
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            errors.append(rel_err)
    return np.concatenate(errors)


def _worst_error(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute worst relative error over ensemble."""
    errs = _all_relative_errors(nodes, weights, w0_vals, wa_vals, z_vals)
    return float(np.max(errs))


def _smooth_max_error(nodes, weights, w0_vals, wa_vals, z_vals, beta=50.0):
    """Smooth approximation of max error using log-sum-exp."""
    errs = _all_relative_errors(nodes, weights, w0_vals, wa_vals, z_vals)
    # Log-sum-exp for stability
    m = np.max(errs)
    if m < 1e-30:
        return 0.0
    lse = m + np.log(np.sum(np.exp(beta * (errs - m)))) / beta
    return float(lse)


def _params_to_nw(params):
    """Convert parameter vector to (nodes, weights) with constraints."""
    n = 8
    x = params[:n]
    y = params[n:]
    # Nodes: cumulative sum of exponentials for strictly increasing in (0,1)
    ex = np.exp(x - np.max(x))
    cumsum = np.cumsum(ex)
    nodes_p = cumsum / cumsum[-1]
    eps = 1e-12
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i-1]:
            nodes_p[i] = nodes_p[i-1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    # Weights: exp parameterization for positive weights summing to 1
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    return nodes_p, weights_p


def _nw_to_params(nodes, weights):
    """Convert (nodes, weights) back to parameter vector."""
    n = 8
    # Inverse of cumulative exp: log of differences
    diffs = np.diff(nodes, prepend=0.0)
    x = np.log(np.maximum(diffs, 1e-15))
    y = np.log(np.maximum(weights, 1e-15))
    return np.concatenate([x, y])


def _optimize_smooth(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize using smooth surrogate with L-BFGS-B."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _smooth_max_error(nodes_p, weights_p, w0_vals, wa_vals, z_vals, beta=50.0)
    
    # Initial params
    p0 = _nw_to_params(best_nodes, best_weights)
    
    t_start = time.time()
    try:
        res = minimize(
            _objective, p0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-12, 'maxfun': 300, 'ftol': 1e-14}
        )
        if _deadline_ok():
            np_, wp_ = _params_to_nw(res.x)
            err_ = _worst_error(np_, wp_, w0_vals, wa_vals, z_vals)
            if err_ < best_err:
                best_nodes, best_weights, best_err = np_, wp_, err_
    except Exception:
        pass
    
    # Polish with Nelder-Mead on the exact (nonsmooth) objective
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.5:
        def _objective_exact(params):
            if not _deadline_ok():
                return 1e10
            nodes_p, weights_p = _params_to_nw(params)
            return _worst_error(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
        
        p_cur = _nw_to_params(best_nodes, best_weights)
        try:
            res2 = minimize(
                _objective_exact, p_cur,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                         'maxfev': 100}
            )
            if _deadline_ok():
                np2, wp2 = _params_to_nw(res2.x)
                err2 = _worst_error(np2, wp2, w0_vals, wa_vals, z_vals)
                if err2 < best_err:
                    best_nodes, best_weights, best_err = np2, wp2, err2
        except Exception:
            pass
    
    return best_nodes, best_weights


def _multi_start_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Multi-start optimization with diverse initializations."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    # Generate diverse starting points
    starts = [(nodes_init.copy(), weights_init.copy())]
    
    # Perturbed Gauss-Legendre
    rng = np.random.RandomState(42)
    for _ in range(3):
        nodes_p = nodes_init + rng.randn(n) * 0.02
        nodes_p = np.sort(np.clip(nodes_p, 1e-4, 1.0 - 1e-4))
        # Ensure strict increase
        for i in range(1, n):
            if nodes_p[i] <= nodes_p[i-1]:
                nodes_p[i] = nodes_p[i-1] + 1e-4
        weights_p = weights_init * (1.0 + rng.randn(n) * 0.1)
        weights_p = np.maximum(weights_p, 1e-6)
        weights_p = weights_p / weights_p.sum()
        starts.append((nodes_p, weights_p))
    
    # Try each start
    for i, (nd, wt) in enumerate(starts):
        if not _deadline_ok():
            break
        elapsed = time.time() - _T0
        remaining = time_budget - elapsed
        if remaining < 10.0:
            break
        try:
            nd_opt, wt_opt = _optimize_smooth(nd, wt, w0_vals, wa_vals, z_vals, remaining * 0.8)
            err_opt = _worst_error(nd_opt, wt_opt, w0_vals, wa_vals, z_vals)
            if err_opt < best_err:
                best_nodes, best_weights, best_err = nd_opt, wt_opt, err_opt
        except Exception:
            pass
    
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
    
    # Ensemble for optimization
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 80.0
    t_start = time.time()
    
    try:
        nodes, weights = _multi_start_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
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
