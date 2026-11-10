"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s kill
    
    def E_of_a(a, w0, wa):
        """H(a)/H0 for flat LCDM with CPL dark energy: E^2 = Omega_m a^-3 + Omega_de a^(-3(1+w0+wa*(1-a)))"""
        Omega_m = 0.3
        Omega_de = 1.0 - Omega_m
        w = w0 + wa * (1.0 - a)
        E2 = Omega_m * a**(-3) + Omega_de * a**(-3*(1.0 + w))
        return np.sqrt(E2)
    
    def integrand(a, w0, wa):
        """f(a) = a * E(a)"""
        return a * E_of_a(a, w0, wa)
    
    def reference_integral(z, w0, wa, n_points=2000):
        """Adaptive Simpson-like reference using many points"""
        a = 1.0 / (1.0 + z * np.linspace(0, 1, n_points))
        f = integrand(a, w0, wa)
        # Simpson's rule
        h = a[1] - a[0]
        # a is decreasing, so flip
        a_rev = a[::-1]
        f_rev = f[::-1]
        h = a_rev[1] - a_rev[0]
        n = len(a_rev) - 1
        if n % 2 == 1:
            # even number of intervals
            result = h/3 * (f_rev[0] + f_rev[-1] + 4*np.sum(f_rev[1:-1:2]) + 2*np.sum(f_rev[2:-1:2]))
        else:
            # odd number of intervals, use trapezoid for last interval
            result = h/3 * (f_rev[0] + f_rev[-3] + 4*np.sum(f_rev[1:-3:2]) + 2*np.sum(f_rev[2:-3:2]))
            result += h/2 * (f_rev[-2] + f_rev[-1])
        return result
    
    def quadrature_value(z, nodes, weights, w0, wa):
        a_i = 1.0 / (1.0 + z * nodes)
        f_vals = integrand(a_i, w0, wa)
        return z * np.sum(weights * f_vals)
    
    def relative_error(z, nodes, weights, w0, wa):
        ref = reference_integral(z, w0, wa)
        quad = quadrature_value(z, nodes, weights, w0, wa)
        if abs(ref) < 1e-15:
            return 0.0
        return abs(quad - ref) / abs(ref) * 100.0  # percent
    
    # Test cosmologies: diverse w0, wa combinations
    test_models = []
    for w0 in [-2.0, -1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]:
        for wa in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
            test_models.append((w0, wa))
    
    test_z_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Parameterization: 
    # nodes: use sigmoid to map from unconstrained to (0,1) strictly increasing
    # weights: use softmax to ensure positive and sum to 1
    
    def decode(params):
        """Decode unconstrained params to valid nodes and weights"""
        # First 8 params -> nodes via sigmoid, then enforce ordering
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Sigmoid to get values in (0,1)
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params * 10.0))
        # Enforce strictly increasing: sort and add small epsilon separation
        sorted_idx = np.argsort(raw_nodes)
        ordered = raw_nodes[sorted_idx]
        # Ensure strict increase by adding small offsets
        eps = 1e-6
        for i in range(1, 8):
            if ordered[i] <= ordered[i-1]:
                ordered[i] = ordered[i-1] + eps
        # Clip to (0,1)
        ordered = np.clip(ordered, eps, 1.0 - eps)
        
        # Softmax for weights
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        return ordered, weights
    
    def objective(params):
        nodes, weights = decode(params)
        max_err = 0.0
        for w0, wa in test_models:
            for z in test_z_values:
                err = relative_error(z, nodes, weights, w0, wa)
                if err > max_err:
                    max_err = err
        return max_err
    
    from scipy.optimize import minimize
    
    # Initial guess: midpoint rule
    init_nodes = (np.arange(8) + 0.5) / 8.0
    init_weights = np.full(8, 1.0/8.0)
    
    # Encode initial
    # For nodes: inverse sigmoid
    init_node_params = np.log(init_nodes / (1.0 - init_nodes)) / 10.0
    # For weights: log
    init_weight_params = np.log(init_weights)
    
    x0 = np.concatenate([init_node_params, init_weight_params])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # L-BFGS-B optimization
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.fun < best_score:
        best_params = result.x
        best_score = result.fun
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        x_rand = np.random.randn(16) * 2.0
        res = minimize(objective, x_rand, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    
    nodes, weights = decode(best_params)
    
    # Final cleanup: ensure strict constraints
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
