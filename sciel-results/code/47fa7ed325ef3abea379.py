import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # CPL model: E(a) = a^p * exp(q*(1-a)) where p=1.5*(w0+w1-1), q=1.5*w1
    # We need universal 8-node quadrature for integral of 1/E(a(t)) over t in [0,1]
    # where a(t) = 1/(1+z*t)

    # Design basis: diverse cosmologies to span the parameter space
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5]
    w1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_vals = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

    basis_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                basis_cases.append((z, w0, w1))

    n_cases = len(basis_cases)

    def _compute_exact(z, w0, w1):
        """Compute exact integral ∫_0^1 dt/E(a(t)) using adaptive quad"""
        p = 1.5 * (w0 + w1 - 1.0)
        q = 1.5 * w1

        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            # E(a) = a^p * exp(q*(1-a))
            # 1/E(a) = a^(-p) * exp(-q*(1-a))
            log_val = -p * np.log(a) - q * (1.0 - a)
            if log_val > 700:
                return 1e300
            if log_val < -700:
                return 0.0
            return np.exp(log_val)

        res, _ = si.quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
        return res

    # Precompute exact values
    exact_vals = np.zeros(n_cases)
    for i, (z, w0, w1) in enumerate(basis_cases):
        if time.time() > deadline - 5.0:
            exact_vals[i:] = 1.0
            break
        try:
            exact_vals[i] = _compute_exact(z, w0, w1)
        except Exception:
            exact_vals[i] = 1.0

    # Ensure no zeros
    exact_vals = np.maximum(exact_vals, 1e-30)

    # Parameterization:
    # params = [x1, y1, y2, ..., y7, logw1, logw2, ..., logw8]
    # x1: logit of first node
    # yj: log of gap between consecutive logits (guarantees strictly increasing)
    # logw: log weights (normalized to sum to 1)
    # Total: 8 + 7 + 8 = 23 parameters... wait, 1 + 7 + 8 = 16

    def _param_to_nodes_weights(params):
        """Convert 16-dim param vector to (nodes, weights) with guaranteed constraints"""
        x1 = params[0]
        y = params[1:8]  # 7 log-gaps
        logw = params[8:16]  # 8 log-weights

        # Build logits: x[0] = x1, x[i] = x[i-1] + exp(y[i-1])
        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i - 1] + np.exp(y[i - 1])

        # Convert to nodes in (0,1)
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)

        # Convert to weights
        w = np.exp(logw - np.max(logw))
        w = w / np.sum(w)

        return nodes, w

    def _quadrature_value(nodes, weights, z, w0, w1):
        """Compute quadrature estimate of ∫_0^1 dt/E(a(t))"""
        p = 1.5 * (w0 + w1 - 1.0)
        q = 1.5 * w1
        a_vals = 1.0 / (1.0 + z * nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_inv_E = -p * np.log(a_vals) - q * (1.0 - a_vals)
        log_inv_E = np.clip(log_inv_E, -700, 700)
        inv_E = np.exp(log_inv_E)
        return np.dot(weights, inv_E)

    def _relative_errors(nodes, weights):
        """Compute relative errors for all basis cases"""
        errs = np.zeros(n_cases)
        for i, (z, w0, w1) in enumerate(basis_cases):
            exact_val = exact_vals[i]
            if exact_val <= 0:
                continue
            quad_val = _quadrature_value(nodes, weights, z, w0, w1)
            errs[i] = (quad_val - exact_val) / exact_val
        return errs

    def _residuals(params):
        """Residuals for least_squares: relative errors"""
        nodes, weights = _param_to_nodes_weights(params)
        return _relative_errors(nodes, weights)

    def _minmax_objective(params):
        """Minimize worst-case absolute relative error"""
        nodes, weights = _param_to_nodes_weights(params)
        errs = _relative_errors(nodes, weights)
        return np.max(np.abs(errs))

    # Initial guess: Gauss-Legendre on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Convert to parameter space
    x_init = np.log(gl_nodes / (1.0 - gl_nodes))
    params_init = np.zeros(16)
    params_init[0] = x_init[0]
    for i in range(1, 8):
        gap = x_init[i] - x_init[i - 1]
        if gap <= 0:
            gap = 1e-6
        params_init[1 + i - 1] = np.log(gap)
    params_init[8:16] = np.log(gl_weights)

    best_params = params_init.copy()
    best_cost = np.sum(np.square(_relative_errors(*_param_to_nodes_weights(params_init))))
    best_minmax = _minmax_objective(params_init)

    # Stage 1: Least-squares to get a good starting point
    try:
        res_ls = so.least_squares(_residuals, params_init, method='trf',
                                  max_nfev=5000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if res_ls.cost < best_cost:
            best_params = res_ls.x
            best_cost = res_ls.cost
    except Exception:
        pass

    # Stage 2: Min-max Nelder-Mead to reduce worst-case error
    if time.time() < deadline - 15.0:
        try:
            res_mm = so.minimize(_minmax_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 8000, 'xatol': 1e-12,
                                          'fatol': 1e-14, 'adaptive': True})
            if res_mm.fun < best_minmax:
                best_minmax = res_mm.fun
                best_params = res_mm.x
        except Exception:
            pass

    # Stage 3: Another least-squares pass from min-max result (refinement)
    if time.time() < deadline - 10.0:
        try:
            res_ls2 = so.least_squares(_residuals, best_params, method='trf',
                                       max_nfev=3000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if res_ls2.cost < best_cost:
                best_cost = res_ls2.cost
                # Only accept if min-max also improves or stays similar
                mm_check = _minmax_objective(res_ls2.x)
                if mm_check <= best_minmax * 1.05:
                    best_params = res_ls2.x
                    best_minmax = min(best_minmax, mm_check)
        except Exception:
            pass

    # Stage 4: Final min-max polish
    if time.time() < deadline - 5.0:
        try:
            res_mm2 = so.minimize(_minmax_objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-13,
                                           'fatol': 1e-15, 'adaptive': True})
            if res_mm2.fun < best_minmax:
                best_minmax = res_mm2.fun
                best_params = res_mm2.x
        except Exception:
            pass

    # Extract final nodes and weights
    final_nodes, final_weights = _param_to_nodes_weights(best_params)

    # Final cleanup: ensure strict constraints
    final_nodes = np.sort(np.clip(final_nodes, 1e-10, 1.0 - 1e-10))
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i - 1]:
            final_nodes[i] = final_nodes[i - 1] + 1e-10
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)

    return {"nodes": final_nodes, "weights": final_weights}
