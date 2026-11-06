import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    Omega_m = 0.3
    Omega_L = 0.7

    # Define a diverse but manageable set of test cases
    # z: redshift, w0: dark energy equation of state, wa: time variation
    z_vals = np.array([0.1, 0.3, 0.7, 1.5, 3.0, 6.0])
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])
    wa_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))
    
    # Add some extreme cases to ensure robustness
    test_cases.append((10.0, -1.0, 0.0))
    test_cases.append((10.0, 0.0, 0.0))
    test_cases.append((0.05, -2.0, 2.0))
    test_cases.append((0.05, 1.0, -2.0))

    n_tests = len(test_cases)
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    # Precompute high-accuracy reference values using 64-point Gauss-Legendre
    # 64 points is sufficient for high accuracy in these smooth integrals
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    ref_vals = np.zeros(n_tests)
    for i in range(n_tests):
        z, w0, wa = z_arr[i], w0_arr[i], wa_arr[i]
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        
        # E(a)^2 = Omega_m/a^3 + Omega_L * exp(3(1+w0+wa)ln(a) - 3wa(a-1))
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        ref_vals[i] = np.dot(gl_weights, inv_E)

    def _compute_quadrature_error(nodes, weights):
        """
        Compute the worst-case relative error in percent for the given nodes and weights.
        """
        # nodes: shape (8,)
        # weights: shape (8,)
        
        # a_vals[i, j] = 1 / (1 + z_i * nodes_j)
        # Shape: (n_tests, 8)
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        
        # Compute 1/E(a) for all test cases and nodes
        # log_terms shape: (n_tests, 8)
        log_terms = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                   - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_terms = np.clip(log_terms, -700, 700)
        
        term_L = Omega_L * np.exp(log_terms)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E  # Shape (n_tests, 8)
        
        # Quadrature values: sum_j weights[j] * inv_E[i, j]
        # Using matrix multiplication: (n_tests, 8) @ (8,) -> (n_tests,)
        qvals = inv_E @ weights
        
        # Relative error
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    def _param_to_nodes_weights(params):
        """
        Convert parameter vector to nodes and weights.
        params:
        - first 7: log-gaps between nodes
        - 8th: log-first-node (mapped to (0,1) via sigmoid)
        - next 8: log-weights
        """
        # Nodes
        log_gaps = params[:7]
        log_first = params[7]
        
        # First node via sigmoid to ensure (0, 1)
        x0 = 1.0 / (1.0 + np.exp(-log_first))
        
        # Gaps via softplus or exp to ensure positive
        # Using exp with bounds on log_gaps ensures positive gaps
        gaps = np.exp(log_gaps)
        
        x = np.zeros(8)
        x[0] = x0
        for i in range(7):
            x[i+1] = x[i] + gaps[i]
            
        # Ensure nodes are strictly increasing and within (0,1)
        # If they go out of bounds, clip and adjust
        # A better way: normalize if they exceed 1, or clip
        # For simplicity and stability, we'll clip and enforce strict increase
        nodes = np.clip(x, 1e-8, 1.0 - 1e-8)
        # Enforce strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-8
        nodes[7] = min(nodes[7], 1.0 - 1e-8)
        
        # Weights
        log_weights = params[8:16]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        return nodes, weights

    def _objective(params):
        nodes, weights = _param_to_nodes_weights(params)
        return _compute_quadrature_error(nodes, weights)

    # Initial guess: Gauss-Legendre 8-point rule
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0
    
    # Convert to parameter space for initialization
    # Invert the mapping
    params_init = np.zeros(16)
    
    # Nodes
    # Approximate log_first from x0
    x0 = gl8_nodes[0]
    params_init[7] = np.log(x0 / (1.0 - x0))
    
    # Gaps
    for i in range(7):
        gap = gl8_nodes[i+1] - gl8_nodes[i]
        if gap < 1e-12:
            gap = 1e-12
        params_init[i] = np.log(gap)
        
    # Weights
    params_init[8:16] = np.log(np.maximum(gl8_weights, 1e-12))

    # Bounds
    # log_gaps: reasonable range for gaps in [0,1]
    # If nodes span [0,1], gaps are on average 1/7 ~ 0.14
    # log(0.01) ~ -4.6, log(0.5) ~ -0.7
    bounds = [(-5.0, 0.5)] * 7  # log_gaps
    bounds.append((-10.0, 10.0)) # log_first (sigmoid input)
    bounds.extend([(-10.0, 10.0)] * 8) # log_weights

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = _compute_quadrature_error(best_nodes, best_weights)

    # Global Optimization: Differential Evolution
    # DE is good for non-smooth, multimodal problems
    if time.time() < deadline - 20.0:
        try:
            # Use a moderate population size and max iterations
            # polluting the search space with random restarts is handled by DE internally
            res = so.differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mut=(0.5, 1.0),
                recomb=0.7,
                polish=False, # Polish can be slow and DE often finds good enough
                update_interval=1, # Update best each iteration
                callback=lambda x, convergence: None
            )
            
            # Check result
            if res.fun < best_err:
                best_err = res.fun
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
        except Exception as e:
            pass

    # Local Refinement: L-BFGS-B
    # Refine the best solution found by DE using gradient-based method
    # This can squeeze out more accuracy
    if time.time() < deadline - 5.0:
        try:
            # Re-initialize params from current best
            params_local = np.zeros(16)
            x0 = best_nodes[0]
            params_local[7] = np.log(x0 / (1.0 - x0))
            for i in range(7):
                gap = best_nodes[i+1] - best_nodes[i]
                if gap < 1e-12: gap = 1e-12
                params_local[i] = np.log(gap)
            params_local[8:16] = np.log(np.maximum(best_weights, 1e-12))
            
            # Tighter bounds for local search? Keep same bounds for safety
            res_local = so.minimize(
                _objective,
                params_local,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res_local.fun < best_err:
                best_err = res_local.fun
                best_nodes, best_weights = _param_to_nodes_weights(res_local.x)
        except Exception:
            pass

    # Final Cleanup
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
