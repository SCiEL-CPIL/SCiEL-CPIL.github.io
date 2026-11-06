"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    Omega_m = 0.3
    Omega_L = 0.7

    # Dense grid for optimization
    # We want to cover the expected range of validation cosmologies.
    # z: 0.1 to 5.0 (log spaced?)
    # w0: -1.5 to 0.5
    # wa: -1.5 to 1.5
    
    z_grid = np.logspace(-1, np.log10(5.0), 10)
    w0_grid = np.linspace(-1.5, 0.5, 7)
    wa_grid = np.linspace(-1.5, 1.5, 7)
    
    test_cases = []
    for z in z_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                test_cases.append((z, w0, wa))
                
    # Fast reference integration using high-order Gauss-Legendre
    # 64 points is very accurate for these smooth functions
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(64)
    ref_nodes = (ref_nodes + 1.0) / 2.0
    ref_weights = ref_weights / 2.0

    def _fast_integral(z, w0, wa):
        a_vals = 1.0 / (1.0 + z * ref_nodes)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = term_m + term_L
        E2 = np.maximum(E2, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(ref_weights, inv_E)

    # Precompute reference values using vectorized operations over the grid
    # z_grid, w0_grid, wa_grid are 1D. We need to expand them to match test_cases structure.
    # test_cases is a list of tuples. Let's convert to arrays for vectorization.
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])
    
    # Compute reference values for all test cases at once
    # a_vals shape: (num_cases, 64)
    # ref_nodes shape: (64,)
    a_ref = 1.0 / (1.0 + np.outer(z_arr, ref_nodes))
    a_ref = np.clip(a_ref, 1e-12, 1.0)
    
    log_term_ref = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_ref) - 3.0 * wa_arr[:, None] * (a_ref - 1.0)
    log_term_ref = np.clip(log_term_ref, -700, 700)
    term_L_ref = Omega_L * np.exp(log_term_ref)
    term_m_ref = Omega_m * (a_ref ** -3)
    E2_ref = term_m_ref + term_L_ref
    E2_ref = np.maximum(E2_ref, 1e-300)
    E_ref = np.sqrt(E2_ref)
    inv_E_ref = 1.0 / E_ref
    ref_vals = np.dot(inv_E_ref, ref_weights)

    def _param_to_nodes_weights(params):
        # params: [x1, y1..y7, logw1..logw8]
        x1 = params[0]
        y = params[1:8]
        logw = params[8:16]

        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i-1] + np.exp(y[i-1])

        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)

        w = np.exp(logw)
        w = w / np.sum(w)

        return nodes, w

    def _objective(params):
        # Check time limit
        if time.time() > deadline - 2.0:
            return 1e10
            
        nodes, weights = _param_to_nodes_weights(params)
        
        # Vectorized quadrature calculation for all test cases
        # a_vals shape: (num_cases, 8)
        a_q = 1.0 / (1.0 + np.outer(z_arr, nodes))
        a_q = np.clip(a_q, 1e-12, 1.0)
        
        log_term_q = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_q) - 3.0 * wa_arr[:, None] * (a_q - 1.0)
        log_term_q = np.clip(log_term_q, -700, 700)
        term_L_q = Omega_L * np.exp(log_term_q)
        term_m_q = Omega_m * (a_q ** -3)
        E2_q = term_m_q + term_L_q
        E2_q = np.maximum(E2_q, 1e-300)
        E_q = np.sqrt(E2_q)
        inv_E_q = 1.0 / E_q
        
        # q_vals shape: (num_cases,)
        q_vals = np.dot(inv_E_q, weights)
        
        # Relative errors
        rel_errs = np.abs(q_vals - ref_vals) / ref_vals
        
        # Use Sum of Squared Errors (L2) for smoother optimization landscape
        # This is often easier to minimize than L-infinity (max error)
        return np.sum(rel_errs ** 2)

    # Initial guess: Gauss-Legendre
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0

    x_init = np.log(nodes_init / (1.0 - nodes_init))
    params_init = np.zeros(16)
    params_init[0] = x_init[0]
    for i in range(1, 8):
        gap = x_init[i] - x_init[i-1]
        if gap <= 0:
            gap = 1e-6
        params_init[1+i-1] = np.log(gap)
    params_init[8:16] = np.log(weights_init)

    best_params = params_init
    best_cost = _objective(params_init)

    # Optimization
    # Use Nelder-Mead with a tighter budget and fewer restarts to ensure completion.
    # L-BFGS-B might be faster if gradients are available, but Nelder-Mead is derivative-free and robust.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb initial guess for restarts
        if r == 0:
            current_params = params_init
        else:
            # Use a smaller perturbation to stay in a good basin
            current_params = best_params + np.random.normal(0, 0.05, 16)
            
        try:
            res = so.minimize(_objective, current_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except Exception:
            pass

    final_nodes, final_weights = _param_to_nodes_weights(best_params)

    # Final cleanup
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)

    return {"nodes": final_nodes, "weights": final_weights}

# EVOLVE-BLOCK-END
