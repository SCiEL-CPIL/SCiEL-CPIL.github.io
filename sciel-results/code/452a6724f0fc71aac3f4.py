"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Structural change: Direct minimax (Chebyshev) optimization of worst-case relative
    error across a representative CPL cosmology grid, with constraint-guaranteeing
    parameterization and multi-restart search.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s kill
    
    # Representative CPL cosmologies
    Om_m = 0.3
    Om_de = 0.7
    
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.3, 0.0, 0.3, 0.5]
    wa_values = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    
    cosmologies = []
    for w0 in w0_values:
        for wa in wa_values:
            # Skip unphysical or problematic combinations
            if abs(w0 + wa) < 0.05:
                continue
            cosmologies.append((w0, wa))
    
    z_test = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        a = np.asarray(a, dtype=float)
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3.0 * (1.0 + w0))
        else:
            exponent = 3.0 * wa / (1.0 + w0 + wa)
            base = 1.0 + wa * (1.0 - a)
            # Guard against negative base raised to non-integer power
            base = np.maximum(base, 1e-15)
            de_term = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * base**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=20001):
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
    
    def quadrature_value(nodes, weights, z, w0, wa):
        """Compute quadrature approximation of the integral."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return np.sum(weights / E_vals)
    
    # Pre-compute reference values for all test cases
    ref_data = []  # (z, w0, wa, ref_val)
    for w0, wa in cosmologies:
        for z in z_test:
            try:
                rv = reference_integral(z, w0, wa)
                if rv > 1e-12:
                    ref_data.append((z, w0, wa, rv))
            except Exception:
                pass
    
    if not ref_data:
        # Fallback to Gauss-Legendre
        gl_nodes, gl_weights = roots_legendre(8)
        nodes = 0.5 * (gl_nodes + 1.0)
        weights = 0.5 * gl_weights
        return {"nodes": nodes, "weights": weights}
    
    # Parameterization:
    # Nodes: t_i > 0, nodes = cumsum(t) / cumsum(t)[-1], clipped to (eps, 1-eps)
    # Weights: w = exp(y) / sum(exp(y))
    
    def decode_params(params):
        """Decode raw parameters to (nodes, weights) with guaranteed constraints."""
        t = np.exp(params[:8])
        cumsum_t = np.cumsum(t)
        total = cumsum_t[-1]
        nodes = cumsum_t / total
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        
        y = params[8:16]
        y_shifted = y - np.max(y)
        w_exp = np.exp(y_shifted)
        weights = w_exp / np.sum(w_exp)
        
        return nodes, weights
    
    def encode_nodes_weights(nodes, weights):
        """Encode (nodes, weights) to raw parameter space."""
        # For nodes: t_i such that cumsum(t)/total = nodes
        # t_0 = nodes[0]
        # t_i = nodes[i] - nodes[i-1] for i > 0
        t = np.zeros(8)
        t[0] = nodes[0]
        for i in range(1, 8):
            t[i] = nodes[i] - nodes[i - 1]
        t = np.maximum(t, 1e-10)
        nodes_raw = np.log(t)
        
        y = np.log(weights + 1e-15)
        y = y - np.max(y)
        
        return np.concatenate([nodes_raw, y])
    
    def objective(params):
        """Minimax: return max relative error across all test cases."""
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = decode_params(params)
        
        # Quick validity check
        if not np.all(np.diff(nodes) > 1e-10):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        
        max_err = 0.0
        for z, w0, wa, rv in ref_data:
            try:
                quad_val = quadrature_value(nodes, weights, z, w0, wa)
                err = abs(quad_val - rv) / rv
                if err > max_err:
                    max_err = err
                    if max_err > 1.0:
                        return max_err
            except Exception:
                return 1e10
        
        return max_err
    
    # Initial guesses: Gauss-Legendre and a few perturbations
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
    gl_weights_01 = 0.5 * gl_weights
    
    best_params = encode_nodes_weights(gl_nodes_01, gl_weights_01)
    best_obj = objective(best_params)
    
    # Multi-restart search
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            # First restart: start from GL
            params_init = best_params.copy()
        else:
            # Subsequent restarts: perturb GL
            perturbation_scale = 0.05 * (r + 1)
            params_init = encode_nodes_weights(gl_nodes_01, gl_weights_01)
            noise = rng.normal(0, perturbation_scale, size=16)
            params_init = params_init + noise
        
        try:
            result = minimize(
                objective, params_init, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Try Powell refinement if time allows
    if time.time() < deadline - 10:
        try:
            result_p = minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 50, 'maxfev': 3000, 'ftol': 1e-10}
            )
            if result_p.fun < best_obj:
                best_obj = result_p.fun
                best_params = result_p.x.copy()
        except Exception:
            pass
    
    # Decode final parameters
    nodes, weights = decode_params(best_params)
    
    # Final sanity checks
    if not np.all(np.diff(nodes) > 0) or not np.all(nodes > 0) or not np.all(nodes < 1):
        nodes = gl_nodes_01.copy()
        weights = gl_weights_01.copy()
    
    if not np.all(weights > 0):
        weights = gl_weights_01.copy()
    
    weights = weights / np.sum(weights)
    
    # Ensure sum is exactly 1 within tolerance
    weights = weights / np.sum(weights)
    
    if time.time() > deadline:
        # Emergency fallback
        return {"nodes": gl_nodes_01, "weights": gl_weights_01}
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
