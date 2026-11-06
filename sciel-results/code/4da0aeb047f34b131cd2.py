"""Improved search-based approach with vectorized evaluation and global optimization."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # Denser test grid to better approximate the worst-case over the continuous space
    z_vals = np.logspace(-1, np.log10(5.0), 12)
    w0_vals = np.linspace(-1.5, 0.5, 6)
    wa_vals = np.linspace(-1.5, 1.5, 6)

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))
    
    # Convert to arrays for vectorization
    z_arr = np.array([c[0] for c in test_cases])
    w0_arr = np.array([c[1] for c in test_cases])
    wa_arr = np.array([c[2] for c in test_cases])
    
    N_tests = len(test_cases)

    # Reference integration using high-order Gauss-Legendre (64 points)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    def _compute_references():
        """Compute reference integrals for all test cases."""
        refs = np.zeros(N_tests)
        for i in range(N_tests):
            z = z_arr[i]
            w0 = w0_arr[i]
            wa = wa_arr[i]
            a_vals = 1.0 / (1.0 + z * gl_nodes)
            a_vals = np.clip(a_vals, 1e-15, 1.0)
            log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
            log_term = np.clip(log_term, -700, 700)
            term_L = Omega_L * np.exp(log_term)
            term_m = Omega_m * (a_vals ** -3)
            E2 = np.maximum(term_m + term_L, 1e-300)
            E = np.sqrt(E2)
            inv_E = 1.0 / E
            refs[i] = np.dot(gl_weights, inv_E)
        return refs

    ref_vals = _compute_references()

    def _param_to_nodes_weights(params):
        """Parameterize nodes and weights to ensure constraints."""
        x = params[:8]
        logw = params[8:16]

        # Map x to strictly increasing nodes in (0, 1) using cumulative sum of exponentials
        exp_x = np.exp(x)
        cumsum = np.cumsum(exp_x)
        total = cumsum[-1]
        nodes = cumsum / total
        # Shift slightly away from 0 and 1 to avoid singularities
        nodes = nodes * 0.999 + 0.0005
        
        w = np.exp(logw)
        w = w / np.sum(w)
        
        return nodes, w

    def _objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = _param_to_nodes_weights(params)
        
        # Vectorized quadrature evaluation
        a_vals = 1.0 / (1.0 + np.outer(z_arr, nodes))
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        
        log_term = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        
        q_vals = inv_E.dot(weights)
        
        rel_errs = np.abs(q_vals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    # Run Differential Evolution for global optimization
    bounds = [(-10, 10)] * 16
    
    best_result = None
    best_score = float('inf')
    
    try:
        de_result = so.differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            popsize=20,
            polish=False
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass

    if best_result is not None:
        nodes_out, weights_out = _param_to_nodes_weights(best_result)
    else:
        # Fallback to Gauss-Legendre if optimization fails
        nodes_out, weights_out = np.polynomial.legendre.leggauss(8)
        nodes_out = (nodes_out + 1.0) / 2.0
        weights_out = weights_out / 2.0

    # Final cleanup
    nodes_out = np.clip(nodes_out, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if nodes_out[i] <= nodes_out[i-1]:
            nodes_out[i] = nodes_out[i-1] + 1e-12
    weights_out = np.clip(weights_out, 1e-12, None)
    weights_out = weights_out / np.sum(weights_out)

    return {"nodes": nodes_out, "weights": weights_out}
