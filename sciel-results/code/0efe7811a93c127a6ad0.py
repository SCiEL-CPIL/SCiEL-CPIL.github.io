"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _E_of_a(a, Om, Or, Ode, w0, wa):
    """Compute E(a) = H(a)/H0 for flat CPL dark energy model."""
    # E^2 = Om/a^3 + Or/a^4 + Ode * a^(3*(1+w0-wa)) * exp(3*wa*(a-1))
    term_m = Om / a**3
    term_r = Or / a**4
    # Dark energy term
    exponent = 3.0 * (1.0 + w0 - wa) * np.log(a) + 3.0 * wa * (a - 1.0)
    term_de = Ode * np.exp(exponent)
    E2 = term_m + term_r + term_de
    return np.sqrt(np.maximum(E2, 1e-300))

def _exact_dL(z, Om, Or, Ode, w0, wa):
    """Compute exact dL/(c/H0) via adaptive quadrature."""
    from scipy.integrate import quad
    def integrand(u):
        a = 1.0 / (1.0 + z * u)
        E = _E_of_a(a, Om, Or, Ode, w0, wa)
        return 1.0 / E
    val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, Om, Or, Ode, w0, wa):
    """Compute quadrature approximation of dL/(c/H0)."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, Om, Or, Ode, w0, wa)
    return (1.0 + z) * z * np.dot(weights, 1.0 / E_vals)

def _relative_error(z, nodes, weights, Om, Or, Ode, w0, wa):
    """Compute relative error for a given cosmology and redshift."""
    exact = _exact_dL(z, Om, Or, Ode, w0, wa)
    approx = _quad_dL(z, nodes, weights, Om, Or, Ode, w0, wa)
    if abs(exact) < 1e-15:
        return 0.0
    return abs(approx - exact) / abs(exact)

def _worst_case_error(params, test_cosmologies, test_zs):
    """Compute worst-case relative error across all test cases."""
    # params: 16 values: 8 log-spaced node params + 8 log-weight params
    # Convert to actual nodes and weights
    log_node_params = params[:8]
    log_weight_params = params[8:]
    
    # Convert log params to strictly increasing nodes in (0,1)
    # Use cumulative sum of exponentials to ensure positivity and ordering
    raw_nodes = np.exp(log_node_params)
    nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Re-normalize to ensure strictly increasing (should be by construction)
    
    # Convert log weight params to positive weights summing to 1
    raw_weights = np.exp(log_weight_params)
    weights = raw_weights / np.sum(raw_weights)
    
    max_err = 0.0
    for (Om, Or, Ode, w0, wa) in test_cosmologies:
        for z in test_zs:
            err = _relative_error(z, nodes, weights, Om, Or, Ode, w0, wa)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Define a diverse set of representative cosmologies
    # (Om, Or, Ode, w0, wa) - flat universe so Om+Or+Ode=1
    test_cosmologies = [
        (0.3, 0.0, 0.7, 0.0, 0.0),       # LCDM
        (0.3, 0.0, 0.7, -0.5, 0.5),      # CPL varying w
        (0.25, 0.0, 0.75, -1.0, 0.0),    # Cosmological constant
        (0.3, 0.0, 0.7, -1.5, 1.0),      # Phantom-like
        (0.3, 0.0, 0.7, 0.5, -0.5),      # Quintessence-like
        (0.2, 0.0, 0.8, -1.0, 0.5),      # Different matter density
        (0.35, 0.0, 0.65, -0.8, 0.3),    # Another CPL
        (0.3, 0.0, 0.7, -1.2, 0.8),      # Strongly varying w
        (0.28, 0.0, 0.72, -0.9, -0.3),   # Negative wa
        (0.3, 0.0, 0.7, 0.0, 1.0),       # w varies from 0 to -1
        (0.3, 0.0, 0.7, -2.0, 2.0),      # Extreme phantom
        (0.3, 0.0, 0.7, 1.0, -2.0),      # Extreme quintessence
    ]
    
    # Test redshifts spanning the range
    test_zs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Initial guess: midpoint rule
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    
    # Convert to log parameter space
    # For nodes: use differences to ensure ordering
    # nodes[i] = sum_{j=0}^{i} exp(log_node_params[j]) / sum_{j=0}^{7} exp(log_node_params[j])
    # Initialize log_node_params so that cumsum(exp) gives midpoint nodes
    raw_init = np.diff(np.concatenate([[0.0], init_nodes, [1.0]]))  # 9 values, first and last are 0
    raw_init = raw_init[1:-1]  # 8 values
    log_node_init = np.log(np.maximum(raw_init, 1e-10))
    
    log_weight_init = np.log(init_weights)
    
    init_params = np.concatenate([log_node_init, log_weight_init])
    
    # Bounds for differential evolution
    # log_node_params: each should be reasonable
    bounds_node = [(-10.0, 5.0)] * 8
    bounds_weight = [(-10.0, 10.0)] * 8
    bounds = bounds_node + bounds_weight
    
    # Run differential evolution
    best_params = init_params.copy()
    best_err = _worst_case_error(init_params, test_cosmologies, test_zs)
    
    try:
        # Use differential evolution with time limit
        def objective(params):
            if time.time() > deadline:
                return 1e10
            return _worst_case_error(params, test_cosmologies, test_zs)
        
        result = differential_evolution(
            objective,
            bounds,
            x0=init_params,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            workers=1
        )
        
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x
            best_err = _worst_case_error(best_params, test_cosmologies, test_zs)
        
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            def obj_for_minimize(params):
                if time.time() > deadline:
                    return 1e10
                return _worst_case_error(params, test_cosmologies, test_zs)
            
            result2 = minimize(
                obj_for_minimize,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result2.fun < best_err:
                best_err = result2.fun
                best_params = result2.x
                best_err = _worst_case_error(best_params, test_cosmologies, test_zs)
                
        except Exception:
            pass
    
    # Convert best params back to nodes and weights
    log_node_params = best_params[:8]
    log_weight_params = best_params[8:]
    
    raw_nodes = np.exp(log_node_params)
    nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    raw_weights = np.exp(log_weight_params)
    weights = raw_weights / np.sum(raw_weights)
    
    # Ensure nodes are strictly increasing (should be by construction)
    # Add tiny epsilon if needed
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
