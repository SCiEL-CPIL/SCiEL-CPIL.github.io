"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _E_cpl(a, w0, wa):
    """E(a) = H(a)/H0 for CPL model in flat universe.
    a = 1/(1+z). E(a) = a^(-3) * ((1+z)^(3(1+w0+wa)) * (1 + wa*z)^(3*wa/wa))
    More carefully: E(z) = (1+z)^3 * (1+z)^(3*wa*z) / (1+wa*z)^(3*wa)
    In terms of a: z = 1/a - 1 = (1-a)/a
    E(a) = a^(-3) * a^(-3*wa*(1/a - 1)) * (1 + wa*(1/a - 1))^(3*wa)
    Let's use the standard form:
    E(z) = (1+z)^3 * (1+z)^(3*wa*z) / (1+wa*z)^(3*wa)
    """
    z = 1.0 / a - 1.0
    # Handle z near 0 carefully
    term1 = (1.0 + z)**3
    # (1+z)^(3*wa*z) = exp(3*wa*z * ln(1+z))
    if abs(z) < 1e-10:
        term2 = 1.0
    else:
        term2 = np.exp(3.0 * wa * z * np.log(1.0 + z))
    denom = (1.0 + wa * z)**(3.0 * wa)
    return term1 * term2 / denom


def _integrand(z_node, w0, wa):
    """The integrand is 1/E(a) where a = 1/(1+z_node)."""
    a = 1.0 / (1.0 + z_node)
    return 1.0 / _E_cpl(a, w0, wa)


def _exact_integral(z_max, w0, wa, n_steps=20000):
    """Compute the exact integral ∫_0^{z_max} dz / E(z) using adaptive Simpson or dense trapezoidal."""
    # Use scipy quad for high accuracy
    from scipy.integrate import quad
    f = lambda z: _integrand(np.array([z]), w0, wa)[0]
    result, _ = quad(f, 0, z_max, limit=500)
    return result


def _quadrature_value(nodes, weights, z_max, w0, wa):
    """Compute the quadrature approximation of ∫_0^{z_max} dz/E(z)."""
    z_eval = z_max * nodes
    vals = _integrand(z_eval, w0, wa)
    return np.sum(weights * vals)


def _relative_error(nodes, weights, z_max, w0, wa):
    """Compute relative error in percent for a given cosmology and redshift."""
    exact = _exact_integral(z_max, w0, wa)
    approx = _quadrature_value(nodes, weights, z_max, w0, wa)
    if abs(exact) < 1e-15:
        return 0.0
    return abs(approx - exact) / abs(exact) * 100.0


def construct():
    """Construct an 8-node positive quadrature rule for luminosity distance."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Validation cosmologies: (w0, wa) pairs
    # Include a range of CPL parameters
    cpl_params = [
        (-1.0, 0.0),    # Cosmological constant
        (-1.0, 0.5),
        (-1.0, -0.5),
        (-0.5, 0.0),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.0, 0.0),     # Matter dominated
        (0.0, 0.5),
        (0.0, -0.5),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.5, -0.5),
    ]
    
    # Redshifts to validate at
    z_max_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    # Precompute exact integrals for all validation cases
    print("Computing exact integrals...")
    exact_values = {}
    for w0, wa in cpl_params:
        for zmax in z_max_values:
            exact_values[(w0, wa, zmax)] = _exact_integral(zmax, w0, wa)
    
    print(f"Computed {len(exact_values)} exact integrals in {time.time()-start_time:.1f}s")
    
    # Initial guess: Gauss-Legendre nodes and weights
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    # Transform from [-1,1] to [0,1]
    nodes_init = 0.5 * (nodes_init + 1.0)
    weights_init = 0.5 * weights_init
    
    # Ensure nodes are strictly increasing (they should be from leggauss)
    nodes_init = np.sort(nodes_init)
    # Ensure weights are positive (they should be from leggauss)
    weights_init = np.maximum(weights_init, 1e-10)
    # Normalize weights to sum to 1
    weights_init = weights_init / np.sum(weights_init)
    
    def worst_case_error(params):
        """Compute worst-case relative error over all validation cases."""
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        
        # Unpack parameters
        # Use a reparameterization to ensure nodes are in (0,1) strictly increasing
        # and weights are positive
        # nodes: 8 values, weights: 8 values
        # Total 16 parameters
        
        # Reparameterize nodes to ensure they're in (0,1) and strictly increasing
        # Use cumulative sum of exponentials for nodes
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: ensure strictly increasing in (0,1)
        # Use softplus-like transformation
        raw_nodes = node_params
        # Map to (0,1) via sigmoid, then ensure ordering
        nodes = 1.0 / (1.0 + np.exp(-10.0 * (raw_nodes)))
        # Ensure strict ordering by adding small increments
        for i in range(1, 8):
            nodes[i] = np.maximum(nodes[i], nodes[i-1] + 1e-8)
        # Clamp to (0,1)
        nodes[0] = np.maximum(nodes[0], 1e-6)
        nodes[7] = np.minimum(nodes[7], 1.0 - 1e-6)
        
        # Weights: ensure positive via exp
        weights = np.exp(weight_params)
        weights = weights / np.sum(weights)
        
        # Compute worst-case error
        max_err = 0.0
        for w0, wa in cpl_params:
            for zmax in z_max_values:
                exact = exact_values[(w0, wa, zmax)]
                z_eval = zmax * nodes
                vals = _integrand(z_eval, w0, wa)
                approx = np.sum(weights * vals)
                if abs(exact) > 1e-15:
                    err = abs(approx - exact) / abs(exact) * 100.0
                    max_err = max(max_err, err)
        
        return max_err
    
    # Initialize parameters from the Gauss-Legendre rule
    # Invert the parameterization
    node_params_init = np.log(nodes_init / (1.0 - nodes_init)) / 10.0
    weight_params_init = np.log(weights_init)
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    print(f"Starting optimization at t={time.time()-start_time:.1f}s")
    
    # Use Nelder-Mead
    best_params = params_init.copy()
    best_error = worst_case_error(best_params)
    print(f"Initial worst-case error: {best_error:.4f}%")
    
    try:
        result = minimize(
            worst_case_error,
            params_init,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_error:
            best_params = result.x
            best_error = result.fun
        print(f"Optimization done at t={time.time()-start_time:.1f}s, error: {best_error:.4f}%")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    nodes = 1.0 / (1.0 + np.exp(-10.0 * node_params))
    for i in range(1, 8):
        nodes[i] = np.maximum(nodes[i], nodes[i-1] + 1e-8)
    nodes[0] = np.maximum(nodes[0], 1e-6)
    nodes[7] = np.minimum(nodes[7], 1.0 - 1e-6)
    
    weights = np.exp(weight_params)
    weights = weights / np.sum(weights)
    
    # Final validation
    final_error = worst_case_error(best_params)
    print(f"Final worst-case error: {final_error:.4f}%")
    print(f"Total time: {time.time()-start_time:.1f}s")
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
