"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Use Gauss-Legendre nodes mapped to [0,1] as a robust starting point.
    2. Perform a fast, safe local optimization of both nodes and weights.
    3. Optimize a smooth surrogate (sum of squared relative errors) over a diverse
       set of CPL cosmologies and redshifts.
    4. Use a parameterization that guarantees strictly increasing nodes in (0,1)
       and positive weights summing to 1.
    5. Guard against numerical issues with deadline checks and fallbacks.
    
    The key structural change from the failing version:
    - Use a simpler, more robust parameterization for nodes.
    - Reduce the number of test cases to keep the objective fast.
    - Use L-BFGS-B with tighter tolerances for speed.
    - Add explicit time guards to prevent timeout.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    # Gauss-Legendre nodes and weights as initial guess
    gl_nodes, gl_weights = roots_legendre(8)
    nodes_gl = 0.5 * (gl_nodes + 1.0)  # Map from [-1,1] to [0,1]
    weights_gl = 0.5 * gl_weights
    
    # Representative CPL cosmologies: (w0, wa) pairs
    # Focused on physically plausible range
    cosmologies = [
        (-1.0, 0.0),   # Cosmological constant
        (-1.0, 0.5),   # Phantom
        (-0.5, 0.0),   # Quintessence
        (0.0, 0.0),    # Matter-dominated (for testing)
        (-1.0, -0.5),  # Another phantom
    ]
    
    # Test redshifts
    z_test = [0.5, 1.0, 2.0]
    
    Om_m = 0.3
    Om_de = 0.7
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        a_safe = np.maximum(a, 1e-10)
        matter = Om_m / a_safe**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a_safe**(-3.0 * (1.0 + w0))
        else:
            exponent = 3.0 * wa / (1.0 + w0 + wa)
            base = 1.0 + wa * (1.0 - a_safe)
            # Handle potential negative base with fractional exponent
            base_safe = np.maximum(base, 1e-10)
            de_term = Om_de * a_safe**(-3.0 * (1.0 + w0 + wa)) * base_safe**exponent
        val = matter + de_term
        return np.sqrt(np.maximum(val, 1e-10))
    
    def reference_integral(z, w0, wa, n_ref=2001):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                        4.0 * np.sum(integrand[1:-1:2]) + 
                        2.0 * np.sum(integrand[2:-1:2]))
        return z * integral
    
    # Pre-compute reference values for all test cases
    ref_data = []
    for w0, wa in cosmologies:
        for z in z_test:
            try:
                rv = reference_integral(z, w0, wa)
                if rv > 1e-12:
                    ref_data.append((w0, wa, z, rv))
            except Exception:
                pass
    
    if not ref_data:
        return {"nodes": nodes_gl, "weights": weights_gl}
    
    # Parameterization:
    # Nodes: use logit transformation. Let x_i be raw params, node_i = sigmoid(x_i)
    # To ensure ordering, we use cumulative sum of exponentials:
    # t_i = exp(x_i), node_i = cumsum(t)[i] / cumsum(t)[-1]
    # This guarantees nodes in (0,1) and strictly increasing.
    #
    # Weights: softmax, w_i = exp(y_i) / sum(exp(y_j))
    
    def decode(params):
        """Decode raw parameters to nodes and weights."""
        x = params[:8]
        y = params[8:16]
        
        # Nodes: cumulative exponential
        t = np.exp(x - np.max(x))  # For numerical stability
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        # Ensure strictly in (0,1)
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        
        # Weights: softmax
        y_shifted = y - np.max(y)
        w_exp = np.exp(y_shifted)
        w_sum = np.sum(w_exp)
        if w_sum < 1e-10:
            weights = np.full(8, 0.125)
        else:
            weights = w_exp / w_sum
        
        return nodes, weights
    
    def encode(nodes, weights):
        """Encode nodes and weights to raw parameters."""
        # Invert node parameterization
        # nodes = cumsum(exp(x)) / cumsum(exp(x))[-1]
        # This is hard to invert directly. Instead, we find x such that
        # the relative spacings match.
        # For simplicity, use the differences:
        # Let d_i = node_i - node_{i-1} for i=1..7, d_0 = node_0
        # Then t_i proportional to d_i
        # x_i = log(d_i)
        
        d = np.zeros(8)
        d[0] = nodes[0]
        for i in range(1, 8):
            d[i] = nodes[i] - nodes[i-1]
        d = np.maximum(d, 1e-10)
        x = np.log(d)
        
        # Weights: log
        y = np.log(np.maximum(weights, 1e-10))
        
        return np.concatenate([x, y])
    
    def objective(params):
        """Objective: sum of squared relative errors."""
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = decode(params)
        
        # Check validity
        if not np.all(np.diff(nodes) > 0):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        
        total_err = 0.0
        for w0, wa, z, rv in ref_data:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = E_of_a(a_vals, w0, wa)
            quad_val = np.sum(weights / E_vals) * z
            
            if rv > 1e-12:
                err = abs(quad_val - rv) / rv
                total_err += err**2
        
        return total_err
    
    # Initial guess: encode Gauss-Legendre
    params_init = encode(nodes_gl, weights_gl)
    
    # Optimize
    best_params = params_init
    best_obj = objective(params_init)
    
    try:
        result = minimize(objective, params_init, method='L-BFGS-B',
                         options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10})
        
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Try a few random restarts if time allows
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb initial guess
            noise = np.random.normal(0, 0.1, 16)
            params_try = params_init + noise
            
            try:
                result = minimize(objective, params_try, method='L-BFGS-B',
                                 options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-10})
                
                if result.fun < best_obj:
                    best_params = result.x
                    best_obj = result.fun
            except Exception:
                pass
    
    # Decode final result
    nodes, weights = decode(best_params)
    
    # Final sanity checks
    if not np.all(np.diff(nodes) > 0):
        nodes = nodes_gl
        weights = weights_gl
    if not np.all(weights > 0):
        weights = weights_gl
    if abs(np.sum(weights) - 1.0) > 1e-8:
        weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    if not np.all(np.diff(nodes) > 0):
        nodes = nodes_gl
        weights = weights_gl
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
