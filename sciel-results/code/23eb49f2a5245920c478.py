"""Basis-function projection quadrature design for luminosity distance."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # High-accuracy reference using 128-point Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(128)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Diverse test set: broader range of z, w0, wa
    z_vals = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    w0_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    wa_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    n_tests = len(test_cases)

    # Precompute reference values vectorized
    ref_vals = np.zeros(n_tests)
    for i, (z, w0, wa) in enumerate(test_cases):
        if time.time() > deadline - 60.0:
            break
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        ref_vals[i] = np.dot(gl_weights, inv_E)

    # Precompute test case arrays for fast vectorized evaluation
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _compute_inv_E_matrix(nodes):
        """Compute 1/E(a) matrix for all test cases at given nodes."""
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_terms = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                   - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_terms = np.clip(log_terms, -700, 700)
        term_L = Omega_L * np.exp(log_terms)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        return 1.0 / E

    def _worst_rel_error(nodes, weights):
        """Worst-case relative error in percent."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    # Start with Gauss-Legendre as initial guess
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = _worst_rel_error(best_nodes, best_weights)

    # Try various node distributions and optimize weights for each
    def _optimize_weights_for_nodes(nodes):
        """Given fixed nodes, find optimal weights by linear least squares."""
        inv_E = _compute_inv_E_matrix(nodes)
        w, residuals, rank, sv = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
        return w

    # Try various node spacings
    node_sets = [
        gl8_nodes.copy(),
        np.linspace(0.05, 0.95, 8),
    ]

    # Beta-distributed nodes (concentrated near endpoints)
    for alpha, beta in [(2, 2), (3, 3), (2, 5), (5, 2), (1.5, 1.5), (4, 1), (1, 4)]:
        from scipy.stats import beta as beta_dist
        quantiles = np.linspace(0.05, 0.95, 8)
        nodes_beta = beta_dist.ppf(quantiles, alpha, beta)
        if np.all(np.diff(nodes_beta) > 1e-8) and np.all(nodes_beta > 0) and np.all(nodes_beta < 1):
            node_sets.append(nodes_beta)

    # Power-law spaced nodes
    for p in [0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
        t = np.linspace(0.05, 0.95, 8)
        nodes_pow = t ** p
        if np.all(np.diff(nodes_pow) > 1e-8):
            node_sets.append(nodes_pow)

    # Log-spaced nodes
    for log_base in [2, 3, 5, 10]:
        t = np.linspace(0.02, 0.98, 8)
        nodes_log = 1.0 / (1.0 + log_base * t)
        nodes_log = np.clip(nodes_log, 0.01, 0.99)
        if np.all(np.diff(nodes_log) > 1e-8):
            node_sets.append(nodes_log)

    # Evaluate each node set with optimal weights
    for nodes in node_sets:
        if time.time() > deadline - 30.0:
            break
        try:
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            # Ensure strictly increasing
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1]:
                    nodes[i] = nodes[i-1] + 1e-6
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)

            w = _optimize_weights_for_nodes(nodes)
            # Ensure positive weights
            if np.any(w <= 0):
                # Fall back to uniform
                w = np.ones(8) / 8.0
            else:
                w = w / np.sum(w)

            err = _worst_rel_error(nodes, w)
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = w.copy()
        except Exception:
            pass

    # Phase 2: Local refinement of nodes and weights
    if time.time() < deadline - 15.0:
        try:
            def _objective(params):
                # params: 7 log-gaps + 1 log-first + 8 log-weights
                log_gaps = params[:7]
                log_first = params[7]
                log_weights = params[8:16]

                gap_sizes = np.exp(log_gaps)
                x = np.zeros(8)
                x[0] = log_first
                for i in range(7):
                    x[i+1] = x[i] + gap_sizes[i]

                nodes = 1.0 / (1.0 + np.exp(-x))
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                for i in range(1, 8):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-10

                w = np.exp(log_weights)
                w = w / np.sum(w)

                return _worst_rel_error(nodes, w)

            # Convert best to parameter space
            x_best = np.log(best_nodes / (1.0 - best_nodes))
            params_init = np.zeros(16)
            params_init[7] = x_best[0]
            for i in range(7):
                gap = x_best[i+1] - x_best[i]
                if gap < 1e-12:
                    gap = 1e-12
                params_init[i] = np.log(gap)
            params_init[8:16] = np.log(np.maximum(best_weights, 1e-12))

            bounds = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8

            res = so.minimize(
                _objective,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if res.fun < best_err:
                best_err = res.fun
                # Extract nodes and weights
                log_gaps = res.x[:7]
                log_first = res.x[7]
                log_weights = res.x[8:16]

                gap_sizes = np.exp(log_gaps)
                x = np.zeros(8)
                x[0] = log_first
                for i in range(7):
                    x[i+1] = x[i] + gap_sizes[i]

                best_nodes = 1.0 / (1.0 + np.exp(-x))
                best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
                for i in range(1, 8):
                    if best_nodes[i] <= best_nodes[i-1]:
                        best_nodes[i] = best_nodes[i-1] + 1e-10

                best_weights = np.exp(log_weights)
                best_weights = best_weights / np.sum(best_weights)
        except Exception:
            pass

    # Phase 3: Final weight optimization for best nodes
    if time.time() < deadline - 5.0:
        try:
            w_final = _optimize_weights_for_nodes(best_nodes)
            if np.all(w_final > 0):
                w_final = w_final / np.sum(w_final)
                err_final = _worst_rel_error(best_nodes, w_final)
                if err_final < best_err:
                    best_err = err_final
                    best_weights = w_final.copy()
        except Exception:
            pass

    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
