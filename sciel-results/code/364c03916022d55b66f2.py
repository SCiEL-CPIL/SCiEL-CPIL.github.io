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

    # Precompute reference values
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

    # Precompute test case arrays
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

    def _soft_max_rel_error(nodes, weights, beta=50.0):
        """Smooth approximation of worst-case relative error using log-sum-exp."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        # Soft-max: (1/beta) * log(sum(exp(beta * errs)))
        max_err = np.max(rel_errs)
        if max_err < 1e-15:
            return 0.0
        scaled = beta * (rel_errs - max_err)
        log_sum_exp = max_err + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
        return log_sum_exp

    # Start with Gauss-Legendre as initial guess
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = _worst_rel_error(best_nodes, best_weights)

    # === BASIS FUNCTION PROJECTION APPROACH ===
    # The integrand 1/E(a) for CPL dark energy can be approximated using
    # physical basis functions. Let t = 1-a (redshift variable).
    # Basis: {1, t, t^2, ..., t^m} plus exponential terms exp(-k*t)
    #
    # For a given cosmology (w0, wa), we fit 1/E(a) ≈ sum_k c_k * phi_k(t)
    # where phi_k are the basis functions. Then the optimal quadrature rule
    # must exactly integrate all basis functions.
    #
    # With 8 nodes and 8 weights, we can exactly integrate 8 basis functions.
    # Choose basis to capture the shape of 1/E(a) across cosmologies.

    def _build_basis_functions(t_values, n_basis=8):
        """Build basis function matrix: powers of t and exponential terms."""
        # t_values: array of shape (n_points,)
        n_pts = len(t_values)
        basis = np.zeros((n_pts, n_basis))
        # Powers of t: t^0, t^1, ..., t^4 (5 basis functions)
        for k in range(5):
            basis[:, k] = t_values ** k
        # Exponential terms: exp(-k*t) for k = 1, 2, 3 (3 basis functions)
        for k in range(3):
            basis[:, 5 + k] = np.exp(-((k + 1) * 2.0) * t_values)
        return basis

    def _fit_basis_coefficients(nodes, w0, wa, z):
        """Fit basis coefficients for a single cosmology."""
        a_vals = 1.0 / (1.0 + z * nodes)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        t_vals = 1.0 - a_vals
        
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        
        basis = _build_basis_functions(t_vals, 8)
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(basis, inv_E, rcond=None)
        return coeffs

    # Try various node distributions and optimize weights for each
    def _optimize_weights_for_nodes(nodes):
        """Given fixed nodes, find optimal weights by linear least squares."""
        inv_E = _compute_inv_E_matrix(nodes)
        w, residuals, rank, sv = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
        return w

    # Generate diverse node sets
    node_sets = [
        gl8_nodes.copy(),
        np.linspace(0.05, 0.95, 8),
    ]

    # Beta-distributed nodes
    for alpha, beta in [(2, 2), (3, 3), (2, 5), (5, 2), (1.5, 1.5), (4, 1), (1, 4), (2, 3), (3, 2)]:
        from scipy.stats import beta as beta_dist
        quantiles = np.linspace(0.05, 0.95, 8)
        nodes_beta = beta_dist.ppf(quantiles, alpha, beta)
        if np.all(np.diff(nodes_beta) > 1e-8) and np.all(nodes_beta > 0) and np.all(nodes_beta < 1):
            node_sets.append(nodes_beta)

    # Power-law spaced nodes
    for p in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
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

    # Chebyshev nodes
    k = np.arange(8)
    nodes_cheb = 0.5 * (1.0 + np.cos(np.pi * (2*k + 1) / (2*8)))
    nodes_cheb = nodes_cheb[::-1]  # Reverse to get increasing order
    nodes_cheb = np.clip(nodes_cheb, 1e-6, 1.0 - 1e-6)
    if np.all(np.diff(nodes_cheb) > 1e-8):
        node_sets.append(nodes_cheb)

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

    # Phase 2: Local refinement with smooth soft-max objective
    if time.time() < deadline - 15.0:
        try:
            def _param_to_nodes_weights(params):
                """Convert parameter vector to nodes and weights."""
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
                
                return nodes, w

            def _objective_smooth(params):
                """Smooth objective using soft-max."""
                nodes, weights = _param_to_nodes_weights(params)
                return _soft_max_rel_error(nodes, weights, beta=50.0)

            def _objective_exact(params):
                """Exact worst-case error for final evaluation."""
                nodes, weights = _param_to_nodes_weights(params)
                return _worst_rel_error(nodes, weights)

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
            
            # First pass with smooth objective
            res = so.minimize(
                _objective_smooth,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11},
            )
            
            if res.fun < best_err * 1.01:  # Allow slight tolerance for smooth vs exact
                nodes_smooth, weights_smooth = _param_to_nodes_weights(res.x)
                err_smooth = _worst_rel_error(nodes_smooth, weights_smooth)
                if err_smooth < best_err:
                    best_err = err_smooth
                    best_nodes = nodes_smooth.copy()
                    best_weights = weights_smooth.copy()
        except Exception:
            pass

    # Phase 3: Epigraph formulation with tighter polish
    if time.time() < deadline - 5.0:
        try:
            def _epigraph_objective(params):
                """Minimize t subject to |q_i - ref_i|/ref_i * 100 <= t for all i."""
                log_gaps = params[:7]
                log_first = params[7]
                log_weights = params[8:15]
                t = params[15]
                
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
                
                inv_E = _compute_inv_E_matrix(nodes)
                qvals = inv_E @ w
                rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
                
                # Violation penalty
                violation = np.max(rel_errs - t)
                if violation > 0:
                    return t + 100.0 * violation
                return t

            # Initialize from current best
            x_best = np.log(best_nodes / (1.0 - best_nodes))
            params_init = np.zeros(16)
            params_init[7] = x_best[0]
            for i in range(7):
                gap = x_best[i+1] - x_best[i]
                if gap < 1e-12:
                    gap = 1e-12
                params_init[i] = np.log(gap)
            params_init[8:15] = np.log(np.maximum(best_weights, 1e-12))
            params_init[15] = best_err  # Initial t
            
            bounds = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 7 + [(0.0, 100.0)]
            
            res = so.minimize(
                _epigraph_objective,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-10},
            )
            
            if res.fun < best_err:
                log_gaps = res.x[:7]
                log_first = res.x[7]
                log_weights = res.x[8:15]
                
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
                
                best_err = _worst_rel_error(best_nodes, best_weights)
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2.0:
        try:
            # Final weight optimization with tighter tolerance
            inv_E = _compute_inv_E_matrix(best_nodes)
            w_final, _, _, _ = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
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
