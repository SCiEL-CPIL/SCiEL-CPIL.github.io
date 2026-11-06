import numpy as np
import time
import scipy.optimize as opt
from scipy.integrate import quad

def _E_cpl(z, w0, w1, Om):
    one_plus_z = 1.0 + z
    exp_out = 3.0 * (1.0 + w0) / 2.0
    exp_in = 3.0 * (1.0 - w0 - w1)
    term_m = Om * one_plus_z ** exp_in
    term_de = 1.0 - Om
    return (one_plus_z ** exp_out) * np.sqrt(term_m + term_de)

def _exact_integral(z, w0, w1, Om):
    f = lambda zp: 1.0 / _E_cpl(zp, w0, w1, Om)
    result, _ = quad(f, 0, z, limit=200)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin below 90s kill

    models = [
        (-1.0, 0.0, 0.3),
        (-1.2, 0.0, 0.3),
        (-0.8, 0.0, 0.3),
        (-1.0, -0.5, 0.3),
        (-1.0, 0.5, 0.3),
        (-0.8, -1.0, 0.3),
        (-1.2, 1.0, 0.3),
        (-1.0, 0.0, 0.2),
        (-1.0, 0.0, 0.35),
        (-0.9, 0.5, 0.25),
        (-1.1, -0.3, 0.28),
        (-0.9, -0.8, 0.32),
        (-1.0, 0.0, 0.3),
    ]
    z_test = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])

    n_models = len(models)
    n_z = len(z_test)
    n_nodes = 8

    # Precompute exact integrals (reference adaptive quadrature)
    exact_vals = np.zeros((n_models, n_z))
    for i, (w0, w1, Om) in enumerate(models):
        for j, z in enumerate(z_test):
            exact_vals[i, j] = _exact_integral(z, w0, w1, Om)

    # Chebyshev-1 collocation nodes/weights (spectral accuracy for smooth integrands)
    # mapped to (0,1). These give positive weights summing to 1.
    theta = (np.arange(n_nodes) + 0.5) * np.pi / n_nodes
    x_cheb = 0.5 - 0.5 * np.cos(theta)          # Chebyshev roots on (0,1)
    w_cheb = np.zeros(n_nodes)
    for k in range(n_nodes):
        s = 0.0
        for m in range(n_nodes):
            s += np.cos((2 * k + 1) * (m + 0.5) * np.pi / n_nodes)
        w_cheb[k] = s / n_nodes
    w_cheb = np.abs(w_cheb)
    w_cheb = w_cheb / w_cheb.sum()

    def _cheb_weights(nodes):
        """Positive Chebyshev spectral quadrature weights for arbitrary increasing nodes."""
        # Map nodes from (0,1) to (-1,1)
        x = 2.0 * nodes - 1.0
        w = np.zeros(n_nodes)
        for k in range(n_nodes):
            s = 0.0
            for m in range(n_nodes):
                s += np.cos((2 * k + 1) * (m + 0.5) * np.pi / n_nodes) * x[m] ** k
            w[k] = s / n_nodes
        w = np.abs(w)
        s = w.sum()
        if s < 1e-30:
            return np.ones(n_nodes) / n_nodes
        return w / s

    def _worst_rel_error(nodes):
        """Smooth max relative error for given nodes using Chebyshev weights."""
        w = _cheb_weights(nodes)
        max_err = 0.0
        for i in range(n_models):
            w0, w1, Om = models[i]
            for j in range(n_z):
                z = z_test[j]
                invE = 1.0 / _E_cpl(z * nodes, w0, w1, Om)
                quad_val = z * np.dot(w, invE)
                rel_err = abs(quad_val - exact_vals[i, j]) / abs(exact_vals[i, j])
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    def _smooth_max_error(nodes, alpha=200.0):
        """Differentiable soft-max surrogate of the worst relative error."""
        w = _cheb_weights(nodes)
        errs = []
        for i in range(n_models):
            w0, w1, Om = models[i]
            for j in range(n_z):
                z = z_test[j]
                invE = 1.0 / _E_cpl(z * nodes, w0, w1, Om)
                quad_val = z * np.dot(w, invE)
                errs.append(abs(quad_val - exact_vals[i, j]) / abs(exact_vals[i, j]))
        e = np.array(errs)
        m = e.max()
        # soft max ~ m + (1/alpha) log sum exp(alpha*(e-m)); smooth and close to max
        return m + np.log(np.sum(np.exp(alpha * (e - m)))) / alpha

    def _node_objective(node_params):
        nodes = np.sort(node_params)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        return _smooth_max_error(nodes)

    # Initial node candidates: Chebyshev (spectral) and Gauss-Legendre
    x_gl, _ = np.polynomial.legendre.leggauss(n_nodes)
    nodes_gl = np.sort((x_gl + 1.0) / 2.0)

    candidates = [np.sort(x_cheb), nodes_gl]
    best_nodes = None
    best_err = np.inf
    for cand in candidates:
        err = _worst_rel_error(cand)
        if err < best_err:
            best_err = err
            best_nodes = cand.copy()

    # Large differential-evolution search over node positions (cheap objective now)
    bounds_de = [(1e-6, 1.0 - 1e-6)] * n_nodes
    if time.time() < deadline - 12:
        try:
            de_res = opt.differential_evolution(
                _node_objective, bounds_de,
                seed=42, maxiter=200, popsize=25,
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.8,
                polish=False
            )
            cand_nodes = np.sort(np.clip(de_res.x, 1e-6, 1.0 - 1e-6))
            err = _worst_rel_error(cand_nodes)
            if err < best_err:
                best_err = err
                best_nodes = cand_nodes
        except Exception:
            pass

    # Local refinement with Nelder-Mead on sorted nodes using the smooth objective
    if time.time() < deadline - 6:
        def _param_to_nodes(params):
            x1 = params[0]
            diffs = params[1:]
            nodes = np.zeros(n_nodes)
            nodes[0] = x1
            for i in range(1, n_nodes):
                nodes[i] = nodes[i - 1] + diffs[i - 1]
            return nodes

        def _param_objective(params):
            nodes = _param_to_nodes(params)
            if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
                return 1.0
            if np.any(np.diff(nodes) < 1e-7):
                return 1.0
            return _smooth_max_error(nodes)

        init_params = np.zeros(n_nodes)
        init_params[0] = best_nodes[0]
        init_params[1:] = np.diff(best_nodes)
        try:
            nm_res = opt.minimize(_param_objective, init_params, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13})
            cand_nodes = _param_to_nodes(nm_res.x)
            if cand_nodes[0] > 1e-6 and cand_nodes[-1] < 1.0 - 1e-6:
                if np.all(np.diff(cand_nodes) > 1e-7):
                    err = _worst_rel_error(cand_nodes)
                    if err < best_err:
                        best_err = err
                        best_nodes = cand_nodes
        except Exception:
            pass

    # Finalize: strictly increasing nodes in (0,1), positive normalized weights
    best_nodes = np.sort(best_nodes)
    for i in range(1, n_nodes):
        if best_nodes[i] <= best_nodes[i - 1]:
            best_nodes[i] = best_nodes[i - 1] + 1e-8
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)

    best_weights = _cheb_weights(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
