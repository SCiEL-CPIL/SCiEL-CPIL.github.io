"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_sq(a, Om, w0, wa):
    """Compute E^2(a) for flat CPL dark energy model."""
    w = w0 + wa * (1.0 - a)
    return Om * a**(-3) + (1.0 - Om) * a**(-3.0 * (1.0 + w))

def _E(a, Om, w0, wa):
    return np.sqrt(_E_sq(a, Om, w0, wa))

def _exact_integral(z, Om, w0, wa):
    """Compute the exact integral using adaptive quadrature (high accuracy)."""
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

def _nodes_from_params(params):
    """Convert 7 params to 8 strictly increasing nodes in (0,1) via cumulative exp."""
    raw = np.exp(params)
    nodes = np.concatenate([[0.0], np.cumsum(raw)])
    nodes = nodes / nodes[-1]
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    return np.clip(nodes, 1e-8, 1.0 - 1e-8)

def _weights_from_params(params):
    """Convert 8 params to positive weights summing to 1 via softmax."""
    w = np.exp(params)
    return w / w.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Diverse cosmological models for training
    cosmo_params = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.3, -0.5, 0.0),
        (0.3, -0.5, 0.5),
        (0.3, -1.5, 0.0),
        (0.3, -1.5, 0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.0, -0.3),
        (0.3, -1.0, 1.0),
        (0.3, -0.8, 0.2),
        (0.3, -1.2, -0.2),
        (0.28, -1.0, 0.4),
        (0.32, -1.0, -0.4),
    ]
    
    z_list = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 0.2, 0.8, 1.2, 2.5]
    
    # Pre-compute exact values
    exact_vals = []
    for z, (Om, w0, wa) in zip(z_list, cosmo_params):
        exact_vals.append(_exact_integral(z, Om, w0, wa))
    
    def _objective(params):
        node_params = params[:7]
        weight_params = params[7:15]
        nodes = _nodes_from_params(node_params)
        weights = _weights_from_params(weight_params)
        
        max_rel_err = 0.0
        for i, (z, (Om, w0, wa)) in enumerate(zip(z_list, cosmo_params)):
            exact = exact_vals[i]
            approx = _quadrature_value(nodes, weights, z, Om, w0, wa)
            if abs(exact) > 1e-10:
                rel_err = abs(approx - exact) / abs(exact) * 100.0
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        return max_rel_err
    
    # Multiple starting points
    n = 8
    
    # Start 1: midpoint rule
    init_nodes_1 = (np.arange(n) + 0.5) / n
    init_weights_1 = np.full(n, 1.0 / n)
    # Convert to param space: node_params from log of differences
    node_diffs_1 = np.diff(np.concatenate([[0], init_nodes_1, [1]]))
    # We have 9 diffs, use first 7 as params (last 2 determined by normalization)
    # Actually, our _nodes_from_params uses 7 params -> 8 nodes via cumsum
    # So we need to map init_nodes to 7 params
    # nodes[0] = 0, nodes[1] = exp(p0), ..., nodes[7] = sum(exp(p0..p6))
    # After normalization: nodes[i] = cumsum(exp(p))[:i+1] / total
    # This is complex to invert. Let's just use random starts.
    
    # Gauss-Legendre mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_mapped = (gl_nodes + 1.0) / 2.0
    gl_weights_mapped = gl_weights / 2.0
    
    # Create param vectors for starting points
    def _nodes_to_params(nodes):
        """Convert 8 nodes to 7 params for our parameterization."""
        # nodes[i] = cumsum(exp(p))[:i+1] / total
        # This is hard to invert analytically. Use a simple approach:
        # Just return log of node differences as a rough approximation
        diffs = np.diff(np.concatenate([[0], nodes, [1]]))
        # Take first 7 diffs
        return np.log(np.maximum(diffs[:7], 1e-10))
    
    def _weights_to_params(weights):
        """Convert 8 weights to 8 params (log)."""
        return np.log(np.maximum(weights, 1e-10))
    
    start_points = []
    
    # Midpoint
    start_points.append(np.concatenate([
        _nodes_to_params(init_nodes_1),
        _weights_to_params(init_weights_1)
    ]))
    
    # Gauss-Legendre
    start_points.append(np.concatenate([
        _nodes_to_params(gl_nodes_mapped),
        _weights_to_params(gl_weights_mapped)
    ]))
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        rand_nodes = np.sort(rng.uniform(0.05, 0.95, 8))
        rand_weights = rng.dirichlet(np.ones(8))
        start_points.append(np.concatenate([
            _nodes_to_params(rand_nodes),
            _weights_to_params(rand_weights)
        ]))
    
    best_obj = np.inf
    best_params = None
    
    for sp in start_points:
        if time.time() > deadline - 15.0:
            break
        
        obj_val = _objective(sp)
        if obj_val < best_obj:
            best_obj = obj_val
            best_params = sp.copy()
        
        try:
            result = minimize(_objective, sp, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract nodes and weights from best solution
    if best_params is not None:
        nodes = _nodes_from_params(best_params[:7])
        weights = _weights_from_params(best_params[7:15])
    else:
        nodes = init_nodes_1
        weights = init_weights_1
    
    # Ensure validity
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    
    # Final check
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(weights.sum() - 1.0) > 1e-12:
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
