import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))
    
    def _true_distance(z_max, w0, w1):
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val
    
    # Use a structured test grid
    w0_grid = np.array([-1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    w1_grid = np.array([-0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    z_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    
    test_cases = []
    for w0 in w0_grid:
        for w1 in w1_grid:
            for z in z_grid:
                test_cases.append((w0, w1, z))
    
    n_cases = len(test_cases)
    true_vals = np.zeros(n_cases)
    for i, (w0, w1, z) in enumerate(test_cases):
        if time.time() > deadline - 15:
            break
        true_vals[i] = _true_distance(z, w0, w1)
    
    def _objective_from_nodes_weights(nodes, weights):
        """Compute worst-case relative error."""
        max_err = 0.0
        for i, (w0, w1, z) in enumerate(test_cases):
            if true_vals[i] == 0:
                continue
            z_nodes = z * nodes
            a_nodes = 1.0 / (1.0 + z_nodes)
            vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
            approx = z * np.dot(weights, vals)
            err = abs(approx - true_vals[i]) / true_vals[i] * 100.0
            if err > max_err:
                max_err = err
        return max_err
    
    # Parameterization: use log-spaced gaps to ensure monotonicity
    # node_0 = exp(g0) / (1 + exp(g0))  -- no, let's use a simpler transform
    # Let delta_i > 0, node_0 = delta_0, node_i = node_{i-1} + delta_i
    # All nodes in (0,1) means sum of deltas < 1
    # Use: nodes = cumulative sum of sigmoid(params) * scale
    
    def _params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights."""
        # First 7 params define gaps between nodes (7 gaps for 8 nodes)
        # Last 8 params define weights (unconstrained, will be normalized)
        gaps_raw = params[:7]
        weights_raw = params[7:15]
        
        # Transform gaps to positive values in (0, 0.5) to ensure sum < 1
        gaps = 0.5 * np.tanh(gaps_raw) + 0.25  # range (0.25, 0.75), too wide
        # Better: gaps = exp(gaps_raw) / (1 + exp(gaps_raw)) * 0.4 + 0.01
        gaps = 0.4 / (1.0 + np.exp(-gaps_raw)) + 0.01  # range (0.01, 0.41)
        
        # Ensure total sum < 0.99
        total = np.sum(gaps)
        if total > 0.99:
            gaps = gaps * 0.99 / total
        
        nodes = np.zeros(8)
        nodes[0] = gaps[0]
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + gaps[i]
        
        # Weights: softmax to ensure positive
        weights = np.exp(weights_raw - np.max(weights_raw))
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        return _objective_from_nodes_weights(nodes, weights)
    
    # Initial guess from Gauss-Legendre
    t, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w_gl
    weights_init = weights_init / np.sum(weights_init)
    
    # Invert to get initial params
    gaps_init = np.diff(np.concatenate([[0], nodes_init]))[1:]  # 7 gaps
    gaps_init = np.log((gaps_init - 0.01) / 0.4 * (1 - (gaps_init - 0.01) / 0.4))
    weights_log_init = np.log(weights_init)
    
    x0 = np.concatenate([gaps_init, weights_log_init])
    
    best_params = x0.copy()
    best_err = _objective(x0)
    
    # Multi-start local optimization
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Perturb initial point
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.5, 15)
        
        try:
            result = minimize(_objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if time.time() < deadline and result.fun < best_err:
                best_params = result.x.copy()
                best_err = result.fun
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 3:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < deadline and result.fun < best_err:
                best_params = result.x.copy()
                best_err = result.fun
        except Exception:
            pass
    
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Final safety checks
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if nodes[j] <= nodes[j-1]:
            nodes[j] = nodes[j-1] + 1e-8
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
