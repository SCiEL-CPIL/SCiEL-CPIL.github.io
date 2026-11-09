"""Rewritten program for luminosity-distance quadrature optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution


def _E_cpl(a, w0, wa):
    """CPL dark energy model for flat universe."""
    Om_m = 0.3
    Om_r = 5e-5
    Om_DE = 1.0 - Om_m - Om_r
    w = w0 + wa * (1 - a)
    return np.sqrt(Om_m * a**(-3) + Om_r * a**(-4) + Om_DE * a**(-3*(1+w0+wa)) * (1 + wa*(1-a)))


def _integrand(z, w0, wa):
    """The integrand for luminosity distance: z / E(a) where a = 1/(1+z)"""
    a = 1.0 / (1.0 + z)
    return z / _E_cpl(a, w0, wa)


def _true_dL(z, w0, wa):
    """Compute true dimensionless luminosity distance using adaptive quadrature."""
    val, _ = quad(_integrand, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1 + z) * val


def _quadrature_dL(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of dimensionless luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(weights / E_vals)
    return z * integral


def _worst_rel_error(nodes, weights, test_zs, test_cosmos):
    """Compute worst relative error (in percent) over test set."""
    max_err = 0.0
    for (w0, wa) in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quadrature_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val * 100.0
                if rel_err > max_err:
                    max_err = rel_err
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 8
    
    # Start from Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1)
    gl_weights = 0.5 * gl_weights
    
    # Define test cosmologies and redshifts for optimization
    test_cosmos = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Cosmological constant
        (0.5, -1.0),     # Quintessence-like
        (-1.5, 0.5),     # Phantom-like
        (-0.5, 0.5),     # Moderate evolution
        (0.3, -0.3),     # Another quintessence
        (-1.2, 0.3),     # Phantom with evolution
        (-0.8, -0.2),    # Another model
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    def _params_to_nodes_weights(params):
        """Convert parameters to valid nodes and weights."""
        # First n-1 nodes are free, last node is determined by ordering
        raw_nodes = params[:n]
        # Use softplus-like transformation to ensure nodes in (0,1) and increasing
        # Shift and scale to ensure positive, then normalize
        shifted = raw_nodes + 5.0
        # Use cumulative sum of exponentials to ensure strictly increasing
        exp_shifted = np.exp(shifted)
        cumsum = np.cumsum(exp_shifted)
        nodes = cumsum / cumsum[-1]
        # Ensure strictly in (0,1)
        nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
        # Ensure strictly increasing
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
        
        # Weights: use softplus to ensure positive
        raw_weights = params[n:]
        weights = np.exp(raw_weights)
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    def _objective(params):
        """Objective function: worst relative error."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        return _worst_rel_error(nodes, weights, test_zs, test_cosmos)
    
    # Initial parameters from Gauss-Legendre
    # Invert the transformation to get initial params
    # For nodes: we need cumsum(exp(x_i+5)) / cumsum[-1] = gl_nodes
    # This is complex, so just start from a reasonable initialization
    init_params = np.zeros(2 * n)
    # Initialize nodes part to produce roughly GL nodes
    # Simple approach: start from equal spacing
    init_params[:n] = np.linspace(-3, 3, n)
    init_params[n:] = np.log(gl_weights)
    
    # Run differential evolution
    try:
        bounds = [(-10, 10)] * n + [(-5, 5)] * n
        result = differential_evolution(
            _objective, bounds,
            maxiter=50, popsize=15,
            tol=1e-6, mutation=(0.3, 1.0), recombination=0.8,
            seed=42, polish=False,
            init='latinhypercube'
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            # Try a second round with tighter bounds around the best solution
            best_params = result.x
            tight_bounds = []
            for i in range(n):
                tight_bounds.append((best_params[i] - 2, best_params[i] + 2))
            for i in range(n):
                tight_bounds.append((best_params[n+i] - 1, best_params[n+i] + 1))
            
            result2 = differential_evolution(
                _objective, tight_bounds,
                maxiter=30, popsize=12,
                tol=1e-7, mutation=(0.2, 0.8), recombination=0.7,
                seed=123, polish=False,
                init='latinhypercube'
            )
            if result2.fun < result.fun:
                result = result2
        
        best_params = result.x
        nodes, weights = _params_to_nodes_weights(best_params)
        
    except Exception:
        # Fallback to Gauss-Legendre
        nodes = gl_nodes.copy()
        weights = gl_weights.copy()
        nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
    
    # Final validation
    test_error = _worst_rel_error(nodes, weights, test_zs, test_cosmos)
    
    # Ensure weights sum to 1 within tolerance
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
