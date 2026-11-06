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
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # The integral we are approximating is:
    # dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    # Using the quadrature formula given:
    # dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(z * nodes[i])
    # This is a quadrature rule for integral_0^1 g(t) dt where g(t) = 1/E(z*t)
    # approximated as sum_i weights[i] * g(nodes[i])
    # So we need a quadrature rule for integral_0^1 f(t) dt with 8 nodes and weights.
    
    # The function f(t) = 1/E(z*t) depends on the cosmology model and the redshift z.
    # We want a rule that works well for a variety of such functions.
    
    # Strategy: Model the integrand as a linear combination of basis functions.
    # For CPL models, 1/E(z) can be approximated by functions of the form:
    # (1+z)^(-alpha) * (1 + w0 + wa*z)^(-beta)
    # But more practically, let's use a set of representative models and find
    # nodes/weights that minimize the worst-case error.
    
    # Define E(z) for various models
    def E_lcdm(z):
        Omega_m = 0.3
        Omega_L = 0.7
        return np.sqrt(Omega_m * (1+z)**3 + Omega_L)
    
    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        # E(z)^2 = Omega_m*(1+z)^3 + Omega_DE * (1+z)^(-3*wa) * exp(3*(w0+wa)*z)
        # More stable: E(z)^2 = Omega_m*(1+z)^3 + Omega_DE * exp(3*((w0+wa)*z - wa*ln(1+z)))
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        # Clip to avoid overflow
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))
    
    # Set of test models (w0, wa) pairs
    test_models = [
        (0.3, 0.7, 'lcdm'),  # special case
        (-1.0, 0.0),   # LCDM-like constant w
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
    ]
    
    # Test redshifts
    z_values = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Build E functions
    E_funcs = []
    for m in test_models:
        if len(m) == 3 and m[2] == 'lcdm':
            E_funcs.append(E_lcdm)
        else:
            E_funcs.append(lambda z, w0=m[0], wa=m[1]: E_cpl(z, w0, wa))
    
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
    # Use Gauss-Legendre nodes on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_weights = gl_weights / 2.0
    
    # Convert to unconstrained parameters
    # For nodes: use log of the nodes (approximately)
    init_params = np.concatenate([np.log(init_nodes + 1e-10), np.log(init_weights + 1e-10)])
    
    # Optimize using multiple methods and restarts
    best_params = init_params.copy()
    best_obj = objective(init_params)
    
    # Multiple initializations
    restarts = [
        init_params,
        init_params + np.random.randn(16) * 0.1,
        np.log(np.linspace(0.05, 0.95, 8)) + np.log(init_weights + 1e-10),
        np.log(1.0 - np.linspace(0.05, 0.95, 8)) + np.log(init_weights + 1e-10),
        np.log(np.sort(np.random.uniform(0.05, 0.95, 8))) + np.log(init_weights + 1e-10),
    ]
    
    for i, init_p in enumerate(restarts):
        if time.time() > deadline - 15.0:
            break
        
        try:
            result = minimize(objective, init_p, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Fine-tuning phase with smaller perturbations
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            try:
                perturbed = best_params + np.random.randn(16) * 0.01
                result = minimize(objective, perturbed, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
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
