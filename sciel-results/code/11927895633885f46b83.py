"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Structural change: Instead of optimizing nodes and weights in a bounded space with
    penalty/constraint handling, we use a parameterization that inherently satisfies
    constraints (cumulative-sum for nodes, softmax for weights). This allows the
    optimizer to work in an unbounded space, which is more robust for L-BFGS-B.
    
    Refinement: We use a dense grid of cosmologies and redshifts to pre-compute
    reference values, then minimize a smooth approximation of the worst-case error.
    We also include a final local refinement step using Nelder-Mead on the transformed
    parameters to polish the result.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s limit

    # 1. Initial Gauss-Legendre nodes mapped to [0, 1]
    gl_nodes, gl_weights = roots_legendre(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights

    # 2. Define representative CPL cosmologies
    Om_m = 0.3
    Om_de = 0.7

    # Dense set of (w0, wa) pairs and redshifts for reference
    # We use a wider range to capture worst-case behavior
    w0_values = np.linspace(-1.5, 0.5, 10)
    wa_values = np.linspace(-1.0, 1.0, 9)
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])

    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            # Handle potential numerical issues with negative base if (1+wa*(1-a)) < 0
            base = 1 + wa*(1-a)
            if base < 0:
                # This shouldn't happen for reasonable w0, wa in our range, but just in case
                # CPL is usually defined for a in (0,1] and w0+wa != 0
                # If base is negative and exponent is non-integer, it's complex.
                # We assume parameters keep base positive.
                base = 1e-10
            de_term = Om_de * a**(-3*(1+w0+wa)) * base**exponent
        return np.sqrt(matter + de_term)

    def reference_integral(z, w0, wa, n_ref=2001):
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
        # Fallback if pre-computation fails or takes too long
        weights = weights0 / np.sum(weights0)
        return {"nodes": nodes0, "weights": weights}
    
    ref_vals = np.array(ref_vals)
    n_cases = len(ref_vals)

    # 4. Parameterization functions
    def params_to_nodes_weights(params):
        """
        Convert unbounded parameters to valid nodes and weights.
        params[0:8] -> nodes via cumulative sum of exp
        params[8:16] -> weights via softmax
        """
        # Nodes: t_i = exp(params[i]), nodes = cumsum(t)/cumsum(t)[-1]
        t = np.exp(params[:8] - np.max(params[:8]))  # Shift for stability
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        # Ensure strictly inside (0,1)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Weights: softmax
        w_exp = np.exp(params[8:] - np.max(params[8:]))
        weights = w_exp / np.sum(w_exp)
        
        return nodes, weights

    def nodes_weights_to_params(nodes, weights):
        """
        Invert the parameterization for initialization.
        """
        # For nodes: nodes = cumsum(exp(p))/sum(exp(p))
        # This is tricky to invert exactly, but we can approximate.
        # Let diffs = nodes - [0, nodes[0], ..., nodes[-2]]
        # diffs[i] = exp(p[i]) / sum(exp(p))
        # This doesn't give p directly.
        # Instead, we can use a simpler parameterization for initialization:
        # Just use log(diffs) where diffs are the gaps.
        # Gap 0: nodes[0]
        # Gap i: nodes[i] - nodes[i-1]
        # If we set p[i] = log(gap[i]), then exp(p) = gap, and cumsum(exp(p)) = cumsum(gap) = nodes * total_gap? No.
        # Let's stick to the optimizer starting from a reasonable unbounded point.
        # For GL nodes, the gaps are roughly equal.
        # Let's just use zeros for the node params as a start, or log of the gaps.
        
        gaps = np.zeros(8)
        gaps[0] = nodes[0]
        for i in range(1, 8):
            gaps[i] = nodes[i] - nodes[i-1]
        
        # Ensure positive gaps
        gaps = np.maximum(gaps, 1e-10)
        
        p_nodes = np.log(gaps)
        
        # For weights: softmax(p) = weights
        # p_i - max(p) = log(weights_i) - log(sum(weights)) ? No.
        # softmax(p)_i = exp(p_i) / sum(exp(p_j))
        # If we want softmax(p) = w, then exp(p_i) = c * w_i
        # p_i = log(c) + log(w_i)
        # We can set c=1, so p_i = log(w_i)
        p_weights = np.log(weights + 1e-10)
        
        return np.concatenate([p_nodes, p_weights])

    def compute_node_integrands(nodes, w0, wa, z):
        """Compute 1/E(a_i) at the 8 nodes for given cosmology and redshift."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return 1.0 / E_vals

    def objective(params):
        """
        Minimize worst-case relative error using smooth max approximation.
        params = [node_params (8), weight_params (8)]
        """
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = params_to_nodes_weights(params)
        
        # Check for valid nodes (should be guaranteed by parameterization, but just in case)
        if not np.all(np.diff(nodes) > 0):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
            
        # Compute quadrature estimates for all test cases
        rel_errors = np.zeros(n_cases)
        for i, (w0, wa, z) in enumerate(test_cases):
            node_vals = compute_node_integrands(nodes, w0, wa, z)
            quad_val = z * np.dot(weights, node_vals)
            rel_errors[i] = np.abs(quad_val - ref_vals[i]) / ref_vals[i]
        
        # Smooth max using log-sum-exp
        # max(x_i) ≈ (1/beta) * log(sum(exp(beta * x_i)))
        beta = 50.0
        # Clip errors to avoid overflow in exp
        rel_errors_clipped = np.clip(rel_errors, -100, 100)
        smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * rel_errors_clipped)))
        
        return smooth_max

    # Initial parameters
    # Invert GL nodes and weights to get initial unbounded params
    params0 = nodes_weights_to_params(nodes0, weights0)
    
    # Optimize with L-BFGS-B (no bounds needed due to parameterization, but we can add large bounds for safety)
    bounds = [(-10, 10)] * 16  # Reasonable bounds for log/gaps and log(weights)
    
    best_params = params0
    best_fun = objective(params0)
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        
        if result.fun < best_fun:
            best_params = result.x
            best_fun = result.fun
    except Exception:
        pass
    
    # Refinement with Nelder-Mead if time allows
    if time.time() < deadline - 10:
        try:
            result_nm = minimize(objective, best_params, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if result_nm.fun < best_fun:
                best_params = result_nm.x
                best_fun = result_nm.fun
        except Exception:
            pass

    # Extract and validate
    best_nodes, best_weights = params_to_nodes_weights(best_params)
    
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
        best_weights = best_weights / np.sum(weights0)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
