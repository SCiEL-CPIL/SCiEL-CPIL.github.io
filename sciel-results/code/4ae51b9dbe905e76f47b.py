"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    # om_de0 = 1 - om_m - om_r - om_k
    om_de0 = 1.0 - om_m - om_r - om_k
    # H^2/H0^2 = om_m a^-3 + om_r a^-4 + om_de0 a^(-3(1+w0+wa)) + om_k
    # Note: For flat universe om_k=0 usually, but we allow it.
    # The formula in the task: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This implies the integral is int_0^z (1+z') / E(z') dz'
    # Let x = z'/z, then dz' = z dx.
    # Integral = (1+z) * int_0^1 z * 1/E(z*x) dx ? 
    # Wait, the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This corresponds to the integral: (1+z) * int_0^z dz' / E(z')
    # Substituting z' = z*x, dz' = z dx:
    # (1+z) * int_0^1 z * 1/E(z*x) dx
    # So the integrand for the quadrature over x in [0,1] is f(x) = z * 1/E(z*x) ?
    # No, the formula given is: (1+z)*z * sum w_i / E(a_i).
    # This implies the quadrature approximates int_0^1 1/E(z*x) dx.
    # Because (1+z) * z * [approx int_0^1 1/E(z*x) dx]
    # Let's verify: dL = c/H0 * int_0^z dz' / E(z')
    # Let z' = z*x. dz' = z dx.
    # dL = c/H0 * int_0^1 z dx / E(z*x)
    # So dL/(c/H0) = z * int_0^1 1/E(z*x) dx.
    # The task formula has an extra (1+z).
    # Standard luminosity distance: d_L = (1+z) d_M.
    # d_M = c/H0 * int_0^z dz'/E(z').
    # So d_L/(c/H0) = (1+z) * int_0^z dz'/E(z') = (1+z) * z * int_0^1 dx/E(z*x).
    # So the quadrature is approximating int_0^1 1/E(z*x) dx.
    # The nodes are x_i in [0,1]. a_i = 1/(1+z*x_i).
    # The value to approximate is I(z) = int_0^1 1/E(z*x) dx.
    # The quadrature is Q = sum w_i / E(z*x_i).
    # The error is |Q - I| / I.
    
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_k + om_de0 * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _compute_errors(nodes, weights, models):
    """
    Vectorized computation of relative errors for a set of models.
    nodes: (8,)
    weights: (8,)
    models: list of (z, w0, wa, om_m, om_r, om_k)
    Returns: array of relative errors.
    """
    errors = np.zeros(len(models))
    # Precompute a_nodes for each model? No, z varies.
    # We can vectorize over models and nodes.
    # Shape: (N_models, 8) for nodes
    # z is (N_models, 1)
    
    z_arr = np.array([m[0] for m in models])
    w0_arr = np.array([m[1] for m in models])
    wa_arr = np.array([m[2] for m in models])
    om_m_arr = np.array([m[3] for m in models])
    om_r_arr = np.array([m[4] for m in models])
    om_k_arr = np.array([m[5] for m in models])
    
    # a_i = 1 / (1 + z * x_i)
    # x_i is nodes (8,)
    # z is (N,)
    # a is (N, 8)
    a = 1.0 / (1.0 + z_arr[:, np.newaxis] * nodes[np.newaxis, :])
    
    # E(a) for each
    om_de0_arr = 1.0 - om_m_arr - om_r_arr - om_k_arr
    # a is (N, 8)
    # om_m_arr etc are (N,) -> need to be (N, 1) for broadcasting
    h2 = (om_m_arr[:, np.newaxis] * a**(-3) + 
          om_r_arr[:, np.newaxis] * a**(-4) + 
          om_k_arr[:, np.newaxis] + 
          om_de0_arr[:, np.newaxis] * a**(-3*(1+w0_arr[:, np.newaxis] + wa_arr[:, np.newaxis])))
    
    E = np.sqrt(np.maximum(h2, 1e-30))
    f_vals = 1.0 / E # (N, 8)
    
    # Quadrature values: sum w_i * f_i
    Q = f_vals @ weights # (N,)
    
    # True values: Use high-order Gauss-Legendre (n=64) as reference
    # This is fast enough for N_models ~ 1000
    # GL nodes/weights for [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    
    # a_gl = 1 / (1 + z * gl_nodes)
    a_gl = 1.0 / (1.0 + z_arr[:, np.newaxis] * gl_nodes[np.newaxis, :])
    h2_gl = (om_m_arr[:, np.newaxis] * a_gl**(-3) + 
             om_r_arr[:, np.newaxis] * a_gl**(-4) + 
             om_k_arr[:, np.newaxis] + 
             om_de0_arr[:, np.newaxis] * a_gl**(-3*(1+w0_arr[:, np.newaxis] + wa_arr[:, np.newaxis])))
    E_gl = np.sqrt(np.maximum(h2_gl, 1e-30))
    f_gl = 1.0 / E_gl # (N, 64)
    
    I = f_gl @ gl_weights # (N,)
    
    # Relative error
    # Avoid division by zero
    rel_err = np.abs(Q - I) / np.maximum(I, 1e-15)
    return rel_err

def _loss(params, models):
    """
    Loss function for Differential Evolution.
    params: 16 values (8 nodes, 8 log_weights)
    """
    nodes = params[:8]
    log_weights = params[8:]
    
    # Enforce strict ordering of nodes
    if not np.all(np.diff(nodes) > 1e-6):
        return 1e10
    if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
        return 1e10
        
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    if not np.all(weights > 0):
        return 1e10
        
    # Compute worst-case error
    errs = _compute_errors(nodes, weights, models)
    return np.max(errs)

def construct():
    """
    Constructs an 8-node positive quadrature rule on [0,1] for luminosity distance.
    Uses Differential Evolution to globally optimize nodes and weights.
    """
    n = 8
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Generate a diverse set of training models
    # The validation set is "disjoint public validation cosmologies".
    # We should cover the likely range of parameters.
    # z: 0.1 to 5.0
    # w0: -1.0 to 1.0
    # wa: -1.0 to 1.0
    # om_m: 0.2 to 0.4
    # om_r: 9e-5
    # om_k: 0.0 (flat)
    
    np.random.seed(42)
    N_models = 2000
    
    z_vals = np.random.uniform(0.1, 5.0, N_models)
    w0_vals = np.random.uniform(-1.0, 1.0, N_models)
    wa_vals = np.random.uniform(-1.0, 1.0, N_models)
    om_m_vals = np.random.uniform(0.2, 0.4, N_models)
    om_r_vals = np.full(N_models, 9e-5)
    om_k_vals = np.zeros(N_models)
    
    models = list(zip(z_vals, w0_vals, wa_vals, om_m_vals, om_r_vals, om_k_vals))
    
    # Initial bounds
    # Nodes: (1e-6, 1-1e-6)
    # Log weights: (-10, 10)
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(-10, 10)] * 8
    
    # Strategy:
    # 1. Start with Gauss-Legendre rule as a good initial guess or constraint?
    # Differential Evolution doesn't take an initial guess, it samples randomly.
    # We can use 'init' to provide a population.
    
    # Let's create a population initialized around GL nodes
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    
    # Create a small perturbation around GL
    pop_size = 50
    init_pop = np.zeros((pop_size, 16))
    for i in range(pop_size):
        # 50% of population: GL + noise
        # 50% of population: Random
        if i < pop_size // 2:
            nodes = gl_nodes + np.random.normal(0, 0.05, 8)
            # Ensure ordering
            nodes = np.sort(nodes)
            # Clamp to bounds
            nodes = np.clip(nodes, 1e-5, 1.0 - 1e-5)
            # Force strict increase
            for j in range(1, 8):
                if nodes[j] <= nodes[j-1]:
                    nodes[j] = nodes[j-1] + 1e-4
            if nodes[-1] > 1.0 - 1e-6:
                nodes = nodes * (1.0 - 1e-5) / nodes[-1]
            
            log_w = np.log(gl_weights) + np.random.normal(0, 0.1, 8)
        else:
            nodes = np.sort(np.random.uniform(1e-5, 1.0 - 1e-5, 8))
            for j in range(1, 8):
                if nodes[j] <= nodes[j-1]:
                    nodes[j] = nodes[j-1] + 1e-4
            log_w = np.random.uniform(-5, 5, 8)
        
        init_pop[i, :8] = nodes
        init_pop[i, 8:] = log_w
    
    # Verify init_pop validity
    for i in range(pop_size):
        if not np.all(np.diff(init_pop[i, :8]) > 1e-6):
            init_pop[i, :8] = gl_nodes
            init_pop[i, 8:] = np.log(gl_weights)

    try:
        # Use Differential Evolution
        # maxiter and tol should be tuned for time budget
        # DE can be slow. Let's limit iterations.
        result = differential_evolution(
            _loss,
            bounds,
            args=(models,),
            seed=42,
            maxiter=30,
            tol=1e-8,
            popsize=15,
            init=init_pop,
            polish=True,  # Use Nelder-Mead for final polish
            polish_kwargs={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
        )
        
        if time.time() < deadline - 5.0:
            best_params = result.x
            best_nodes = best_params[:8]
            best_log_weights = best_params[8:]
            best_weights = np.exp(best_log_weights)
            best_weights = best_weights / np.sum(best_weights)
            
            # Final validity check
            if np.all(np.diff(best_nodes) > 1e-6) and np.all(best_weights > 0) and abs(np.sum(best_weights) - 1.0) < 1e-10:
                return {
                    "nodes": best_nodes,
                    "weights": best_weights
                }
            else:
                # Fallback if DE result is invalid
                return {
                    "nodes": gl_nodes,
                    "weights": gl_weights
                }
                
    except Exception as e:
        # Fallback to Gauss-Legendre
        return {
            "nodes": gl_nodes,
            "weights": gl_weights
        }

# EVOLVE-BLOCK-END
