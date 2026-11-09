import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Representative cosmologies: (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 positive)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 negative)
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -1.5, 0.3),   # Strongly evolving
        (0.3, -0.5, -0.3),  # Mild quintessence with evolution
        (0.35, -1.1, 0.2),  # Slightly matter-dominated
    ]
    
    # Test redshifts
    z_test = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0+w1*z))
        # Handle z=0: (1+0)=1, so 1^anything = 1
        return np.sqrt(Om * (1.0 + z)**3 + OL * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Om, w0, w1):
        """Compute the true integral using adaptive quadrature."""
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res
    
    # Precompute true integrals for all test cases
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
    
    # Check time after precomputation
    if time.time() > deadline:
        # Fallback to Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}
    
    def objective(params):
        """
        params: 16 floats
        first 8: node parameters (unconstrained, mapped via sigmoid chain)
        last 8: weight parameters (unconstrained, mapped via softmax)
        """
        node_params = params[:8]
        weight_params = params[8:]
        
        # Map node_params to strictly increasing nodes in (0, 1)
        # Using cumulative sigmoid mapping
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0]))
        for i in range(1, 8):
            step = 1.0 / (1.0 + np.exp(-node_params[i]))
            nodes[i] = nodes[i-1] + step * (1.0 - nodes[i])
        
        # Ensure nodes are strictly inside (0, 1) and increasing
        # The mapping guarantees 0 < u_1 < u_2 < ... < u_8 < 1
        # But let's clamp to be safe
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        
        # Map weight_params to weights summing to 1 and positive
        # Use softmax
        exp_w = np.exp(weight_params - np.max(weight_params))  # for numerical stability
        weights = exp_w / np.sum(exp_w)
        
        # Calculate worst-case relative error
        max_err = 0.0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                # Evaluate E at z * u_i
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
                        
        return max_err
    
    # Start from Gauss-Legendre rule
    n = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    gl_weights = 0.5 * gl_weights
    gl_weights = gl_weights / gl_weights.sum()
    
    # Invert the sigmoid mapping to get initial node_params
    # This is approximate; we'll just start with reasonable values
    # and let the optimizer handle it
    # For simplicity, start with all zeros (which gives roughly uniform spacing)
    initial_params = np.zeros(16)
    
    # Actually, let's try to better initialize by solving for the sigmoid chain
    # that produces gl_nodes
    # This is complex, so let's just start with a reasonable guess
    # and rely on the optimizer
    
    # Run L-BFGS-B optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        best_params = result.x
    except Exception:
        # Fallback to Gauss-Legendre
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    # Check time
    if time.time() > deadline:
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    # Extract best params
    node_params = best_params[:8]
    weight_params = best_params[8:]
    
    # Reconstruct nodes and weights
    nodes = np.zeros(8)
    nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0]))
    for i in range(1, 8):
        step = 1.0 / (1.0 + np.exp(-node_params[i]))
        nodes[i] = nodes[i-1] + step * (1.0 - nodes[i])
    
    # Ensure nodes are strictly inside (0, 1) and increasing
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    
    # Map weight_params to weights
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Final validation
    if not (len(nodes) == 8 and len(weights) == 8):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(np.diff(nodes) > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(weights > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
