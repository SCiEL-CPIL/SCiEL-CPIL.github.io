"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Strategy: Use a weighted least-squares approach to find nodes and weights that
    minimize worst-case error across a representative set of CPL cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Define E(z) for various models
    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))
    
    # Set of test models (w0, wa) pairs - expanded for better coverage
    test_models = [
        (-1.0, 0.0),    # LCDM
        (-0.5, 0.3),
        (-1.5, 0.2),
        (-0.8, 0.4),
        (-1.2, 0.6),
        (-0.3, 0.1),
        (-1.0, 0.8),
        (0.0, 0.0),
        (-2.0, 0.1),
        (-1.0, -0.5),
        (-0.5, -0.3),
        (-1.5, -0.2),
        (-0.8, -0.4),
        (-1.0, 0.3),
        (-0.7, 0.5),
        (-1.3, 0.4),
        (-0.2, 0.2),
        (-1.8, 0.3),
    ]
    
    # Test redshifts - expanded for better coverage
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Build E functions
    E_funcs = []
    for w0, wa in test_models:
        E_funcs.append(lambda z, w0=w0, wa=wa: E_cpl(z, w0, wa))
    
    # Precompute true integrals for all model-redshift combinations
    true_vals = np.zeros((len(E_funcs), len(z_values)))
    for i, E_func in enumerate(E_funcs):
        for j, z in enumerate(z_values):
            integrand = lambda zp: 1.0 / E_func(zp)
            result, _ = quad(integrand, 0, z, limit=200)
            true_vals[i, j] = (1.0 + z) * result
    
    # Quadrature approximation
    def quadrature_approx(nodes, weights, E_func, z):
        z_nodes = z * nodes
        E_vals = E_func(z_nodes)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    # Objective: minimize worst-case relative error
    def objective(params):
        # params: 16 values - 8 nodes (unconstrained) and 8 weights (unconstrained)
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: transform to (0,1) using cumulative sum of exponentials
        exp_nodes = np.exp(node_params)
        nodes = np.cumsum(exp_nodes)
        nodes = nodes / nodes[-1]
        nodes = np.maximum.accumulate(nodes)
        nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
        
        # Ensure strictly increasing
        for k in range(1, 8):
            if nodes[k] <= nodes[k-1]:
                nodes[k] = nodes[k-1] + 1e-12
        
        # Weights: transform to positive and sum to 1
        exp_weights = np.exp(weight_params)
        weights = exp_weights / np.sum(exp_weights)
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for i, E_func in enumerate(E_funcs):
            for j, z in enumerate(z_values):
                approx_val = quadrature_approx(nodes, weights, E_func, z)
                true_val = true_vals[i, j]
                if true_val > 1e-10:
                    rel_err = abs(approx_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        
        return max_rel_err
    
    # Initial guess: Gauss-Legendre-like nodes
    n = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_weights = gl_weights / 2.0
    
    # Convert to unconstrained parameters
    init_params = np.concatenate([np.log(init_nodes + 1e-10), np.log(init_weights + 1e-10)])
    
    # Optimize using multiple methods and restarts
    best_params = init_params.copy()
    best_obj = objective(init_params)
    
    # Multiple initializations with more diverse starting points
    np.random.seed(42)
    restarts = [
        init_params,
        init_params + np.random.randn(16) * 0.1,
        np.log(np.linspace(0.05, 0.95, 8)) + np.log(init_weights + 1e-10),
        np.log(1.0 - np.linspace(0.05, 0.95, 8)) + np.log(init_weights + 1e-10),
        np.log(np.sort(np.random.uniform(0.05, 0.95, 8))) + np.log(init_weights + 1e-10),
        np.log(np.sort(np.random.uniform(0.1, 0.9, 8))) + np.log(np.random.uniform(0.5, 1.5, 8)),
        np.log(np.sort(np.random.uniform(0.02, 0.98, 8))) + np.log(np.random.uniform(0.3, 2.0, 8)),
    ]
    
    for i, init_p in enumerate(restarts):
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = minimize(objective, init_p, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                                     'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Fine-tuning phase with smaller perturbations
    if time.time() < deadline - 15.0:
        for _ in range(5):
            if time.time() > deadline - 8.0:
                break
            try:
                perturbed = best_params + np.random.randn(16) * 0.005
                result = minimize(objective, perturbed, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13,
                                         'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    # Extract best nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    exp_nodes = np.exp(node_params)
    nodes = np.cumsum(exp_nodes)
    nodes = nodes / nodes[-1]
    nodes = np.maximum.accumulate(nodes)
    nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
    
    # Ensure strictly increasing
    for k in range(1, 8):
        if nodes[k] <= nodes[k-1]:
            nodes[k] = nodes[k-1] + 1e-12
    
    exp_weights = np.exp(weight_params)
    weights = exp_weights / np.sum(exp_weights)
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    # Final validation
    if not np.all(np.diff(nodes) > 0) or not np.all(nodes > 0) or not np.all(nodes < 1):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
