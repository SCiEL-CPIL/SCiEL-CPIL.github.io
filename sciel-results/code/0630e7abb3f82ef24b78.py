"""Improved program: smooth-surrogate multi-start optimization for 8-node universal quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 88.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry.
    E(z)^2 = (1+z)^{3(1+w0+wa)} * (1 + wa*z)^{-3*wa}
    """
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving_fast(z, w0, wa, n_quad=500):
    """Exact comoving distance via Gauss-Legendre, vectorized over z array."""
    if np.isscalar(z) or z.ndim == 0:
        z = np.array([z])
    if np.all(z <= 0):
        return np.zeros_like(z)
    
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    # z_vals: (n_z, n_quad)
    z_vals = z[:, None] * 0.5 * (xs[None, :] + 1.0)
    E_vals = _cpl_E(z_vals, w0, wa)
    # exact = 0.5 * z * sum(ws / E_vals, axis=1)
    exact = 0.5 * z * np.sum(ws[None, :] / E_vals, axis=1)
    return exact


def _quad_comoving(z, w0, wa, nodes, weights):
    """Quadrature comoving distance: z * Σ wᵢ/E(z·uᵢ). Vectorized over z array."""
    if np.isscalar(z) or z.ndim == 0:
        z = np.array([z])
    if np.all(z <= 0):
        return np.zeros_like(z)
    z_nodes = np.outer(z, nodes)  # (n_z, 8)
    E_mat = _cpl_E(z_nodes, w0, wa)
    quad = z * np.sum(weights[None, :] / E_mat, axis=1)
    return quad


def _worst_error_smooth(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=400, beta=50.0):
    """Smooth max surrogate of relative errors. Vectorized."""
    n_z = len(z_vals)
    z_arr = np.array(z_vals)
    all_rels = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_fast(z_arr, w0, wa, n_quad)
            quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
            all_rels.append(rel)
    
    all_rels = np.concatenate(all_rels)
    # Smooth max: (1/beta) * log(sum(exp(beta * rel)))
    m = np.max(all_rels)
    if m < 1e-15:
        return m
    smooth_max = m + np.log(np.sum(np.exp(beta * (all_rels - m)))) / beta
    return float(smooth_max)


def _worst_error_exact(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=400):
    """True worst relative error. Vectorized."""
    n_z = len(z_vals)
    z_arr = np.array(z_vals)
    worst = 0.0
    
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_fast(z_arr, w0, wa, n_quad)
            quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
            worst = max(worst, float(np.max(rel)))
    return worst


def _params_to_nw(params):
    """Convert parameter vector to (nodes, weights) with constraints."""
    n = 8
    x = params[:n]
    y = params[n:]
    # Nodes: cumulative logit to ensure (0,1) increasing
    # Use softplus-like: nodes[i] = sigmoid(x[i]) but need ordering
    # Better: use cumulative exp
    ex = np.exp(x - np.max(x))
    cumsum = np.cumsum(ex)
    nodes_p = cumsum / cumsum[-1]
    eps = 1e-14
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i - 1]:
            nodes_p[i] = nodes_p[i - 1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    # Weights: exp parameterization
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    return nodes_p, weights_p


def _nw_to_params(nodes, weights):
    """Convert (nodes, weights) to parameter vector."""
    n = 8
    # Inverse of cumulative exp: differences
    diffs = np.diff(nodes, prepend=0.0)
    diffs = np.maximum(diffs, 1e-14)
    x = np.log(diffs)
    y = np.log(np.maximum(weights, 1e-14))
    return np.concatenate([x, y])


def _optimize_single(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget, beta=50.0):
    """Run one optimization from a starting point."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return nodes_init, weights_init
    
    p0 = _nw_to_params(nodes_init, weights_init)
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error_exact(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error_smooth(nodes_p, weights_p, w0_vals, wa_vals, z_vals, beta=beta)
    
    t_start = time.time()
    
    # Phase 1: Nelder-Mead with smooth objective
    res = minimize(
        _objective, p0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True, 'maxfev': 300}
    )
    
    if _deadline_ok():
        np_, wp_ = _params_to_nw(res.x)
        err_ = _worst_error_exact(np_, wp_, w0_vals, wa_vals, z_vals)
        if err_ < best_err:
            best_nodes, best_weights, best_err = np_, wp_, err_
    
    # Phase 2: L-BFGS-B refinement
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.5:
        p_cur = res.x
        res2 = minimize(
            _objective, p_cur,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-15, 'maxfun': 200}
        )
        if _deadline_ok():
            np2, wp2 = _params_to_nw(res2.x)
            err2 = _worst_error_exact(np2, wp2, w0_vals, wa_vals, z_vals)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
    
    return best_nodes, best_weights


def construct():
    n = 8
    
    # Ensemble for optimization
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5, 0.8]
    wa_vals = [-0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Multiple starting points
    starts = []
    
    # Start 1: Gauss-Legendre on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_gl = 0.5 * (xs + 1.0)
    weights_gl = 0.5 * ws
    nodes_gl = np.clip(nodes_gl, 1e-14, 1.0 - 1e-14)
    for i in range(1, n):
        if nodes_gl[i] <= nodes_gl[i - 1]:
            nodes_gl[i] = nodes_gl[i - 1] + 1e-14
    nodes_gl[-1] = min(nodes_gl[-1], 1.0 - 1e-14)
    weights_gl = weights_gl / weights_gl.sum()
    starts.append((nodes_gl.copy(), weights_gl.copy()))
    
    # Start 2: Equally spaced
    nodes_eq = np.linspace(0.05, 0.95, n)
    weights_eq = np.ones(n) / n
    starts.append((nodes_eq.copy(), weights_eq.copy()))
    
    # Start 3: Chebyshev nodes on [0,1]
    k = np.arange(n)
    nodes_cheb = 0.5 * (1.0 + np.cos(np.pi * (2 * k + 1) / (2 * n)))
    nodes_cheb = nodes_cheb[::-1]  # increasing
    weights_cheb = np.ones(n) / n
    starts.append((nodes_cheb.copy(), weights_cheb.copy()))
    
    # Start 4: Slightly perturbed Gauss-Legendre
    rng = np.random.default_rng(42)
    nodes_p = nodes_gl + rng.normal(0, 0.01, n)
    nodes_p = np.sort(np.clip(nodes_p, 1e-14, 1.0 - 1e-14))
    weights_p = weights_gl + rng.normal(0, 0.02, n)
    weights_p = np.maximum(weights_p, 1e-6)
    weights_p = weights_p / weights_p.sum()
    starts.append((nodes_p.copy(), weights_p.copy()))
    
    # Optimize from each start
    best_nodes = starts[0][0].copy()
    best_weights = starts[0][1].copy()
    best_err = _worst_error_exact(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    time_budget_total = 80.0
    n_starts = len(starts)
    time_per_start = time_budget_total / n_starts
    
    for idx, (ni, wi) in enumerate(starts):
        if not _deadline_ok():
            break
        t_start = time.time()
        remaining = time_budget_total - (time.time() - _T0)
        if remaining < 5.0:
            break
        tp = min(time_per_start, remaining - 2.0)
        
        try:
            nodes_opt, weights_opt = _optimize_single(ni, wi, w0_vals, wa_vals, z_vals, tp)
        except Exception:
            nodes_opt, weights_opt = ni, wi
        
        err = _worst_error_exact(nodes_opt, weights_opt, w0_vals, wa_vals, z_vals)
        if err < best_err:
            best_nodes, best_weights, best_err = nodes_opt, weights_opt, err
    
    # Final local refinement with more iterations
    if _deadline_ok() and (time.time() - _T0) < _DEADLINE - 5.0:
        try:
            nodes_final, weights_final = _optimize_single(
                best_nodes, best_weights, w0_vals, wa_vals, z_vals, 10.0, beta=100.0
            )
            err_final = _worst_error_exact(nodes_final, weights_final, w0_vals, wa_vals, z_vals)
            if err_final < best_err:
                best_nodes, best_weights, best_err = nodes_final, weights_final, err_final
        except Exception:
            pass
    
    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if best_nodes[i] <= best_nodes[i - 1]:
            best_nodes[i] = best_nodes[i - 1] + 1e-12
    best_nodes[-1] = min(best_nodes[-1], 1.0 - 1e-12)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / best_weights.sum()
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
