"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 75
    
    def E_fast(a, Om, w0, wa):
        a = np.asarray(a, dtype=np.float64)
        term_m = Om / a**3
        if abs(1 - Om) < 1e-15:
            term_de = np.zeros_like(a)
        else:
            exponent = -3*(1+w0) - 3*wa*(1-a)
            term_de = (1-Om) * np.exp(exponent)
        return np.sqrt(term_m + term_de)
    
    # Broader set of test cosmologies
    cosmologies = [
        (0.3, -1.0, 0.0), (0.3, -1.0, 1.0), (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5), (0.2, -1.0, 0.0), (0.4, -1.0, 0.0),
        (0.3, -0.8, -0.2), (0.25, -1.2, 0.3), (0.35, -0.7, 0.4),
        (0.28, -1.1, -0.3),
    ]
    z_test = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    
    # Precompute true values
    u_fine = np.linspace(0, 1, 3000)
    true_vals = np.zeros((len(cosmologies), len(z_test)))
    for idx, (Om, w0, wa) in enumerate(cosmologies):
        for j, z in enumerate(z_test):
            a_arr = 1.0 / (1.0 + z * u_fine)
            E_arr = E_fast(a_arr, Om, w0, wa)
            true_vals[idx, j] = np.trapz(u_fine / E_arr, u_fine)
    
    def fast_error(nodes, weights):
        max_err = 0.0
        # a_nodes[i,j] = 1/(1 + z_test[i] * nodes[j])
        a_nodes = 1.0 / (1.0 + z_test[:, None] * nodes[None, :])
        for idx, (Om, w0, wa) in enumerate(cosmologies):
            E_nodes = E_fast(a_nodes, Om, w0, wa)
            quad_vals = np.sum(weights[None, :] / E_nodes, axis=1)
            for j in range(len(z_test)):
                tv = true_vals[idx, j]
                if tv > 1e-10:
                    rel_err = abs(quad_vals[j] - tv) / tv
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err
    
    n = 8
    # Start with Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (gl_nodes + 1)  # map from [-1,1] to [0,1]
    weights_init = gl_weights / 2.0
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = fast_error(nodes_init, weights_init)
    
    def objective(params):
        nodes = np.sort(params[:8])
        nodes = np.clip(nodes, 1e-5, 1-1e-5)
        log_w = params[8:]
        w = np.exp(log_w)
        w = w / np.sum(w)
        if np.any(np.diff(nodes) < 1e-7):
            return 1e6
        return fast_error(nodes, w)
    
    params_init = np.concatenate([nodes_init, np.log(weights_init)])
    
    # Multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        if r == 0:
            p = params_init.copy()
        else:
            p = params_init + np.random.normal(0, 0.05, size=16)
            p[:8] = np.sort(p[:8])
            p[:8] = np.clip(p[:8], 0.01, 0.99)
        
        try:
            result = minimize(objective, p, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_err:
                nodes = np.sort(result.x[:8])
                nodes = np.clip(nodes, 1e-5, 1-1e-5)
                w = np.exp(result.x[8:])
                w = w / np.sum(w)
                if np.all(np.diff(nodes) > 1e-7):
                    best_nodes = nodes
                    best_weights = w
                    best_err = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-6, 1-1e-6)
    for i in range(7):
        if best_nodes[i+1] <= best_nodes[i] + 1e-8:
            best_nodes[i+1] = best_nodes[i] + 1e-6
    best_weights = np.abs(best_weights)
    s = np.sum(best_weights)
    if s > 0:
        best_weights = best_weights / s
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
