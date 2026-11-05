import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Define a diverse set of CPL (Chevallier-Polarski-Linder) models
    # w0, wa pairs covering a broad range
    models = []
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.3, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for w0 in w0_vals:
        for wa in wa_vals:
            models.append((w0, wa))
    # Add some extreme cases
    models.append((-2.0, 2.0))
    models.append((-2.0, -2.0))
    models.append((1.0, -1.0))
    models.append((1.0, 1.0))
    models.append((-1.0, -1.0))
    models.append((-1.0, 1.0))

    def E_of_a(a, w0, wa):
        """E(a) = H(a)/H0 for flat CPL model"""
        Om_m = 0.3
        Om_de = 1.0 - Om_m
        z = 1.0 / a - 1.0
        w = w0 + wa * z
        E2 = (1.0 + z)**3 + Om_de * (1.0 + z)**(3.0 * (1.0 + w))
        return np.sqrt(np.maximum(E2, 1e-30))

    def integrand(z, w0, wa):
        a = 1.0 / (1.0 + z)
        return (1.0 + z) * z / E_of_a(a, w0, wa)

    def reference_integral(w0, wa, n_points=20000):
        """High-accuracy reference using Simpson's rule on fine grid"""
        z = np.linspace(0, 1, n_points)
        f = np.array([integrand(zi, w0, wa) for zi in z])
        # Simpson's rule
        h = 1.0 / (n_points - 1)
        # Ensure even number of intervals
        if (n_points - 1) % 2 != 0:
            n_points -= 1
            z = np.linspace(0, 1, n_points)
            f = np.array([integrand(zi, w0, wa) for zi in z])
            h = 1.0 / (n_points - 1)
        result = h/3.0 * (f[0] + f[-1] + 4*np.sum(f[1:-1:2]) + 2*np.sum(f[2:-1:2]))
        return result

    # Precompute reference integrals
    refs = []
    for (w0, wa) in models:
        refs.append(reference_integral(w0, wa))
    refs = np.array(refs)

    def quadrature_value(nodes, weights, w0, wa):
        a = 1.0 / (1.0 + nodes)
        E_vals = E_of_a(a, w0, wa)
        return np.sum(weights * (1.0 + nodes) * nodes / E_vals)

    def objective(params):
        """
        params: 8 node parameters (unconstrained), 8 weight parameters (unconstrained)
        Transform to ensure nodes in (0,1) strictly increasing and weights > 0 summing to 1
        """
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: use a parameterization that guarantees strict ordering
        # Use cumulative sum of positive increments
        # node_0 = sigmoid(p0) * 0.95
        # node_i = node_{i-1} + sigmoid(p_i) * (1 - node_{i-1}) * 0.95
        # This ensures 0 < node_0 < node_1 < ... < node_7 < 1
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0])) * 0.95
        for i in range(1, 8):
            increment = 1.0 / (1.0 + np.exp(-node_params[i])) * (1.0 - nodes[i-1]) * 0.95
            nodes[i] = nodes[i-1] + increment
        
        # Ensure nodes are strictly in (0, 1)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        # Ensure strict ordering (should already be, but just in case)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
            if nodes[i] >= 1.0:
                nodes[i] = 1.0 - 1e-6
        
        # Weights: softmax to ensure positive, then normalize
        w_raw = np.exp(weight_params - np.max(weight_params))
        weights = w_raw / np.sum(w_raw)
        
        # Compute max relative error across all models
        max_err = 0.0
        for i, (w0, wa) in enumerate(models):
            q_val = quadrature_value(nodes, weights, w0, wa)
            ref_val = refs[i]
            if ref_val > 1e-10:
                rel_err = abs(q_val - ref_val) / abs(ref_val)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    def extract_nodes_weights(params):
        """Extract nodes and weights from parameters"""
        node_params = params[:8]
        weight_params = params[8:16]
        
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0])) * 0.95
        for i in range(1, 8):
            increment = 1.0 / (1.0 + np.exp(-node_params[i])) * (1.0 - nodes[i-1]) * 0.95
            nodes[i] = nodes[i-1] + increment
        
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
            if nodes[i] >= 1.0:
                nodes[i] = 1.0 - 1e-6
        
        w_raw = np.exp(weight_params - np.max(weight_params))
        weights = w_raw / np.sum(w_raw)
        
        return nodes, weights

    # Initial guess: try to set nodes to approximately equally spaced
    # For equal spacing, we need specific sigmoid values
    # node_0 = 1/8 ≈ 0.125, so sigmoid(p0)*0.95 = 0.125 => sigmoid(p0) = 0.1316 => p0 ≈ -1.77
    # For equal spacing, each increment should be 1/8 of remaining
    x0 = np.zeros(16)
    # Approximate initial parameters for equal spacing
    x0[0] = -1.77  # node_0 ≈ 0.125
    for i in range(1, 8):
        x0[i] = 0.0  # sigmoid(0) = 0.5, increment = 0.5 * remaining * 0.95
    x0[8:16] = 0.0  # equal weights

    best_params = x0.copy()
    best_err = objective(x0)

    # Try differential evolution first for global exploration
    if time.time() - t0 < deadline - 30:
        try:
            bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
            de_res = differential_evolution(
                objective, bounds, 
                maxiter=100, popsize=20, tol=1e-8,
                seed=42, polish=False
            )
            if de_res.fun < best_err:
                best_err = de_res.fun
                best_params = de_res.x
        except Exception:
            pass

    # Then try local optimization from multiple starting points
    strategies = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12}},
        {'method': 'Powell', 'options': {'maxiter': 1000, 'ftol': 1e-12}},
    ]
    
    # Generate multiple initial points
    initial_points = [best_params.copy()]
    np.random.seed(42)
    for _ in range(10):
        pert = best_params + 0.5 * np.random.randn(16)
        initial_points.append(pert)
    
    for strat in strategies:
        if time.time() - t0 > deadline - 10:
            break
        for ip in initial_points:
            if time.time() - t0 > deadline - 10:
                break
            try:
                res = minimize(objective, ip, method=strat['method'], options=strat['options'])
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Extract best nodes and weights
    nodes, weights = extract_nodes_weights(best_params)
    
    # Final validation: ensure sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing in [0,1]
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    return {"nodes": nodes, "weights": weights}
