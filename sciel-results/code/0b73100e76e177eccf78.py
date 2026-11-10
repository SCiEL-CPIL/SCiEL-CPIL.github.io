"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Start with Gauss-Legendre nodes.
    2. Optimize both nodes and weights to minimize worst-case relative error
       across a dense grid of CPL cosmologies and redshifts.
    3. Use smooth max (log-sum-exp) approximation for the min-max objective.
    4. Constrain nodes to be strictly increasing in (0,1) and weights positive, sum=1.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    # 1. Gauss-Legendre nodes mapped to [0, 1]
    gl_nodes, gl_weights = roots_legendre(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    
    # 2. Define representative CPL cosmologies
    Om_m = 0.3
    Om_de = 0.7
    
    # Dense set of (w0, wa) pairs and redshifts for reference
    w0_values = np.linspace(-1.2, 0.5, 8)
    wa_values = np.linspace(-0.8, 1.0, 7)
    z_values = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=5001):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                        4*np.sum(integrand[1:-1:2]) + 
                        2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    # 3. Pre-compute reference values for all test cases
    ref_vals = []
    test_cases = []  # list of (w0, wa, z)
    
    for w0 in w0_values:
        for wa in wa_values:
            for z in z_values:
                if time.time() > deadline:
                    break
                try:
                    rv = reference_integral(z, w0, wa)
                    if rv > 1e-12:
                        ref_vals.append(rv)
                        test_cases.append((w0, wa, z))
                except Exception:
                    continue
    
    if not ref_vals:
        weights = 0.5 * gl_weights
        weights = weights / np.sum(weights)
        return {"nodes": nodes0, "weights": weights}
    
    ref_vals = np.array(ref_vals)
    n_cases = len(ref_vals)
    
    def compute_node_integrands(nodes, w0, wa, z):
        """Compute 1/E(a_i) at the 8 nodes for given cosmology and redshift."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return 1.0 / E_vals
    
    def objective(params):
        """
        Minimize worst-case relative error using smooth max approximation.
        params = [nodes (8), weights (8)]
        """
        if time.time() > deadline:
            return 1e10
        
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints softly
        if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Compute quadrature estimates for all test cases
        rel_errors = np.zeros(n_cases)
        for i, (w0, wa, z) in enumerate(test_cases):
            node_vals = compute_node_integrands(nodes, w0, wa, z)
            quad_val = z * np.dot(weights, node_vals)
            rel_errors[i] = np.abs(quad_val - ref_vals[i]) / ref_vals[i]
        
        # Smooth max using log-sum-exp
        # max(x_i) ≈ (1/beta) * log(sum(exp(beta * x_i)))
        beta = 50.0  # Higher beta = sharper max approximation
        max_val = np.max(rel_errors)
        smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * rel_errors)))
        
        return smooth_max
    
    # Initial parameters
    params0 = np.concatenate([nodes0, weights0 / np.sum(weights0)])
    
    # Bounds: nodes in (0,1), weights > 0
    bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-10, 1.0)] * 8
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < 1e-8:
            best_params = result.x
        else:
            best_params = params0
    except Exception:
        best_params = params0
    
    # Extract and validate
    best_nodes = best_params[:8]
    best_weights = best_params[8:]
    
    # Ensure strict ordering and positivity
    if not (np.all(np.diff(best_nodes) > 1e-8) and np.all(best_nodes > 0) and np.all(best_nodes < 1)):
        best_nodes = nodes0
        best_weights = weights0
    
    if not np.all(best_weights > 0):
        best_weights = weights0
    
    # Normalize weights to sum to 1
    best_weights = best_weights / np.sum(best_weights)
    
    # Final sanity checks
    if not np.all(np.diff(best_nodes) > 0):
        best_nodes = nodes0
        best_weights = weights0
    
    if not np.all(best_weights > 0):
        best_weights = weights0
    
    best_weights = best_weights / np.sum(best_weights)
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = weights0
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
