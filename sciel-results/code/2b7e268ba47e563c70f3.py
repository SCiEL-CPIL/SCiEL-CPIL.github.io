"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s limit
    
    # Step 1: Get Gauss-Legendre nodes and weights as initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes_init = np.sort((gl_nodes + 1.0) / 2.0)
    weights_init = gl_weights / 2.0
    
    # Step 2: Define test cosmologies (representative CPL models)
    test_cosmos = [
        (0.30, 0.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.30, -0.5, -0.5),
        (0.30, -1.5, 0.5),
        (0.25, -0.8, 0.3),
        (0.35, -1.2, 0.2),
        (0.28, -0.9, -0.3),
        (0.32, -1.1, 0.4),
    ]
    
    z_vals = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Step 3: Pre-compute all reference integrals ONCE
    def make_E(Om, w0, w1):
        def E(a):
            a = np.asarray(a, dtype=float)
            a_safe = np.maximum(a, 1e-10)
            term_m = Om / a_safe**3
            term_de = (1 - Om) / a_safe**(3 * (1 + w0)) * np.exp(-3 * w1 * (1 - a_safe))
            return np.sqrt(term_m + term_de)
        return E
    
    # Pre-compute reference values: ref_integrals[cosmo_idx][z_idx] = true integral value
    n_cosmos = len(test_cosmos)
    n_z = len(z_vals)
    ref_integrals = np.zeros((n_cosmos, n_z))
    
    for ci, (Om, w0, w1) in enumerate(test_cosmos):
        E = make_E(Om, w0, w1)
        for zi, z in enumerate(z_vals):
            def integrand(u):
                a_val = 1.0 / (1.0 + z * u)
                return 1.0 / E(a_val)
            true_integral, _ = quad(integrand, 0, 1, limit=200)
            ref_integrals[ci, zi] = (1 + z) * z * true_integral
    
    # Pre-compute E values at all (cosmo, z) pairs for fast evaluation
    # E_vals[ci][zi] = array of 8 E values at the nodes (but nodes change, so we can't pre-compute these)
    # Instead, we pre-compute the E function evaluations for each cosmo at arbitrary nodes.
    
    # Step 4: Fast objective function using pre-computed references
    def objective(params):
        t = time.time()
        if t > deadline:
            return None
        
        nodes_p = params[:8]
        weights_p = params[8:]
        
        # Map nodes to (0,1) via tanh
        nodes_mapped = np.sort((np.tanh(nodes_p) + 1.0) / 2.0)
        
        # Map weights via softmax
        weights_mapped = np.exp(weights_p - np.max(weights_p))
        weights_mapped = weights_mapped / np.sum(weights_mapped)
        
        # Compute worst-case relative error using pre-computed references
        max_rel_err = 0.0
        for ci, (Om, w0, w1) in enumerate(test_cosmos):
            E = make_E(Om, w0, w1)
            for zi, z in enumerate(z_vals):
                a_i = 1.0 / (1.0 + z * nodes_mapped)
                E_vals = E(a_i)
                quad_value = (1 + z) * z * np.sum(weights_mapped / E_vals)
                
                true_value = ref_integrals[ci, zi]
                if abs(true_value) > 1e-15:
                    rel_err = abs(quad_value - true_value) / abs(true_value)
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        
        return max_rel_err * 100.0
    
    # Step 5: Optimization
    # Inverse transform for initialization
    nodes_params_init = np.arctanh(np.clip(2 * nodes_init - 1, -0.999, 0.999))
    weights_params_init = np.log(weights_init)
    params_init = np.concatenate([nodes_params_init, weights_params_init])
    
    best_params = params_init.copy()
    best_err = objective(best_params)
    
    if best_err is None:
        return {"nodes": nodes_init, "weights": weights_init}
    
    # Run multiple optimization attempts with different methods and restarts
    methods_to_try = ['Nelder-Mead', 'Powell', 'Nelder-Mead']
    
    for method in methods_to_try:
        if time.time() > deadline - 5.0:
            break
        
        try:
            if method == 'Nelder-Mead':
                result = minimize(
                    objective,
                    params_init,
                    method=method,
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
            else:
                result = minimize(
                    objective,
                    params_init,
                    method=method,
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
                )
            
            if result.fun is not None and result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    nodes_final_raw = np.tanh(best_params[:8])
    nodes_final = np.sort((nodes_final_raw + 1.0) / 2.0)
    
    weights_final_raw = best_params[8:]
    weights_final = np.exp(weights_final_raw - np.max(weights_final_raw))
    weights_final = weights_final / np.sum(weights_final)
    
    # Ensure nodes are strictly in (0,1) and strictly increasing
    nodes_final = np.clip(nodes_final, 1e-6, 1 - 1e-6)
    nodes_final = np.sort(nodes_final)
    for i in range(1, len(nodes_final)):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-10
    
    # Ensure weights sum to 1 within 1e-12
    weights_final = weights_final / np.sum(weights_final)
    
    if time.time() > deadline:
        return {"nodes": nodes_init, "weights": weights_init}
    
    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
