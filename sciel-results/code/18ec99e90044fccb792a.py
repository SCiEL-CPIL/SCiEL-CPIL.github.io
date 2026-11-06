"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _E_sq(a, Om, w0, wa):
    """Compute E^2(a) for flat CPL dark energy model."""
    w = w0 + wa * (1.0 - a)
    return Om * a**(-3) + (1.0 - Om) * a**(-3.0 * (1.0 + w))

def _E(a, Om, w0, wa):
    return np.sqrt(_E_sq(a, Om, w0, wa))

def _exact_integral(z, Om, w0, wa, n_pts=2000):
    """Compute the exact integral using adaptive quadrature (high accuracy)."""
    # Integrand: 1/E(a) where a = 1/(1+z*x), x in [0,1]
    # dL/(c/H0) = (1+z)*z * integral_0^1 dx / E(1/(1+z*x))
    # Actually the formula given: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # So the integral is integral_0^1 dx / E(1/(1+z*x))
    # Let me define f(x) = 1/E(1/(1+z*x))
    
    from scipy.integrate import quad
    def integrand(x):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / _E(a, Om, w0, wa)
    
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E(a_vals, Om, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)

def _objective(params, z_list, cosmo_params):
    """Maximize the worst-case relative error (to minimize it)."""
    n = 8
    # params: [log_nodes (7 free, first is 0.01, last derived), log_weights (8)]
    # Actually let's use a parameterization that ensures positivity and ordering
    # Use: nodes via sigmoid of free params, weights via softmax
    
    # Unpack: 7 node params (for 8 nodes, with boundaries) and 8 weight params
    node_params = params[:7]
    weight_params = params[7:15]
    
    # Convert node params to strictly increasing nodes in (0,1)
    # Use cumulative sum of exponentials
    raw_nodes = np.exp(node_params)
    nodes = np.concatenate([[0.0], np.cumsum(raw_nodes)])
    # Normalize to [0,1]
    nodes = nodes / nodes[-1]
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    
    # Convert weight params to positive weights summing to 1
    weights = np.exp(weight_params)
    weights = weights / weights.sum()
    
    # Compute worst-case relative error
    max_rel_err = 0.0
    for z, (Om, w0, wa) in zip(z_list, cosmo_params):
        exact = _exact_integral(z, Om, w0, wa)
        approx = _quadrature_value(nodes, weights, z, Om, w0, wa)
        if abs(exact) > 1e-10:
            rel_err = abs(approx - exact) / abs(exact) * 100.0  # in percent
            max_rel_err = max(max_rel_err, rel_err)
    
    return max_rel_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Define a diverse set of cosmological models for training
    # (Om, w0, wa) combinations
    cosmo_params = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.5),      # CPL with positive wa
        (0.3, -1.0, -0.5),     # CPL with negative wa
        (0.3, -0.5, 0.0),      # w0 = -0.5
        (0.3, -0.5, 0.5),      # w0 = -0.5, wa = 0.5
        (0.3, -1.5, 0.0),      # w0 = -1.5 (phantom)
        (0.3, -1.5, 0.5),      # phantom with positive wa
        (0.25, -1.0, 0.3),     # lower matter density
        (0.35, -1.0, -0.3),    # higher matter density
        (0.3, -1.0, 1.0),      # large wa
    ]
    
    # Redshifts to test
    z_list = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Initial guess: midpoint rule
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    
    # Convert to parameter space
    # For nodes: use log of differences
    node_diffs = np.diff(np.concatenate([[0], init_nodes, [1]]))
    # node_diffs has 9 elements: [d0, d1, ..., d8] where sum = 1
    # We use 7 free params for the middle nodes
    # Actually, let me use a simpler parameterization
    
    # Use: 7 free params for node positions (sigmoid), 8 for weights (log)
    # Map init_nodes to params: node_params[i] = log(nodes[i+1] - nodes[i]) for i=0..6
    # But we need to be careful with the boundaries
    
    # Simpler: just use the nodes and weights directly with bounds
    # Use scipy minimize with bounds
    
    # Parameter vector: [nodes[0], ..., nodes[7], weights[0], ..., weights[7]]
    # With constraints: 0 < nodes[0] < ... < nodes[7] < 1, weights > 0, sum(weights) = 1
    
    # Use a transformation: 
    # nodes: use 7 params p[0..6], nodes[i] = sigmoid(p[i]) for i=0..7, but ensure ordering
    # This is tricky. Let me use the cumulative exponential approach more carefully.
    
    # Alternative: use 8 node params and 8 weight params, with penalty for violations
    def _penalized_objective(params):
        nodes = params[:8]
        weights = params[8:16]
        
        # Penalty for nodes not in (0,1) or not strictly increasing
        pen = 0.0
        if nodes[0] <= 0 or nodes[-1] >= 1:
            pen += 1000.0
        for i in range(8):
            if nodes[i] <= 0 or nodes[i] >= 1:
                pen += 100.0 * (1.0 - nodes[i]) if nodes[i] < 0 else 100.0 * (nodes[i] - 1.0)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                pen += 1000.0 * (nodes[i-1] - nodes[i])
        
        # Penalty for weights not positive
        for i in range(8):
            if weights[i] <= 0:
                pen += 1000.0 * (-weights[i])
        
        # Penalty for weights not summing to 1
        wsum = weights.sum()
        pen += 100.0 * (wsum - 1.0)**2
        
        if pen > 10.0:
            return pen
        
        # Normalize weights to sum to 1 for the quadrature evaluation
        w_norm = weights / wsum
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for z, (Om, w0, wa) in zip(z_list, cosmo_params):
            exact = _exact_integral(z, Om, w0, wa)
            approx = _quadrature_value(nodes, w_norm, z, Om, w0, wa)
            if abs(exact) > 1e-10:
                rel_err = abs(approx - exact) / abs(exact) * 100.0
                max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err + pen
    
    # Initial point
    x0 = np.concatenate([init_nodes, init_weights])
    
    # Try to optimize
    best_x = x0.copy()
    best_obj = _penalized_objective(x0)
    
    try:
        result = minimize(_penalized_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10.0:
        # Not enough time for more, use what we have
        pass
    else:
        # Try a second optimization from a different starting point
        # Start from Gauss-Legendre-like nodes
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        # Map from [-1,1] to [0,1]
        gl_nodes_mapped = (gl_nodes + 1.0) / 2.0
        gl_weights_mapped = gl_weights / 2.0
        
        x0_2 = np.concatenate([gl_nodes_mapped, gl_weights_mapped])
        obj_2 = _penalized_objective(x0_2)
        
        if obj_2 < best_obj:
            best_obj = obj_2
            best_x = x0_2.copy()
        
        try:
            result2 = minimize(_penalized_objective, x0_2, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_x = result2.x
        except Exception:
            pass
    
    # Extract nodes and weights from best solution
    nodes = best_x[:8]
    weights = best_x[8:16]
    
    # Ensure validity
    # Sort nodes and ensure they're in (0,1)
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    
    # Ensure weights are positive and sum to 1
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    
    # Final check: if something went wrong, fall back to midpoint
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(weights.sum() - 1.0) > 1e-12:
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
