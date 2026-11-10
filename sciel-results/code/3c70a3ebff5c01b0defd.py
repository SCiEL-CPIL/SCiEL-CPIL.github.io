import numpy as np
from scipy.optimize import minimize

def _cpl_e(a, wa0, wa1):
    """
    Compute E(a) = H(a)/H0 for a CPL model.
    Omega_m = 0.3, Omega_L = 0.7.
    E^2(a) = Omega_m / a^3 + Omega_L * (a / (1 + a*wa1))^(3*(1+wa0))
    """
    omega_m = 0.3
    omega_l = 0.7
    
    term_m = omega_m / (a**3)
    base = a / (1.0 + a * wa1)
    base = np.maximum(base, 1e-15)
    exponent = 3.0 * (1.0 + wa0)
    term_l = omega_l * (base ** exponent)
    
    return np.sqrt(term_m + term_l)

def _compute_worst_error(nodes, weights, z_samples, wa0_samples, wa1_samples, n_dense=2000):
    """
    Compute worst-case relative error and all errors for given nodes/weights.
    Returns: (worst_error, all_errors, nodes, weights)
    """
    errors = []
    
    for z in z_samples:
        for wa0 in wa0_samples:
            for wa1 in wa1_samples:
                # Dense reference integration using trapezoidal rule
                u_dense = np.linspace(0, 1, n_dense + 1)
                u_mid = 0.5 * (u_dense[:-1] + u_dense[1:])
                du = u_dense[1] - u_dense[0]
                
                a_mid = 1.0 / (1.0 + z * u_mid)
                E_mid = _cpl_e(a_mid, wa0, wa1)
                integrand = 1.0 / E_mid
                true_val = np.sum(integrand) * du
                
                # Quadrature approximation
                a_quad = 1.0 / (1.0 + z * nodes)
                E_quad = _cpl_e(a_quad, wa0, wa1)
                approx_val = np.sum(weights / E_quad)
                
                if true_val > 1e-12 and approx_val > 1e-12:
                    rel_err = abs(approx_val - true_val) / true_val
                    errors.append(rel_err)
    
    errors = np.array(errors)
    worst_err = np.max(errors) if len(errors) > 0 else 0.0
    return worst_err, errors, nodes, weights

def _objective_logsumsq(params, z_samples, wa0_samples, wa1_samples, n_dense=500):
    """
    Smooth surrogate: sum of log(1 + squared relative errors).
    params: 16 values = [8 node params, 8 weight params]
    """
    # Decode nodes from params: nodes[i] = sigmoid(params[i]) mapped to (0,1)
    # Use tanh to ensure nodes in (0,1) and strictly increasing via sorting
    node_params = params[:8]
    weight_params = params[8:16]
    
    # Map node params to (0,1) using sigmoid, then sort to ensure increasing
    nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
    nodes = np.sort(nodes_raw)
    
    # Ensure strict separation: add small epsilon if needed
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-12:
            nodes[i] = nodes[i-1] + 1e-12
    
    # Clip to (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    # Decode weights: positive via exp, normalize
    weights = np.exp(weight_params)
    weights = weights / np.sum(weights)
    
    # Compute errors
    errors = []
    for z in z_samples:
        for wa0 in wa0_samples:
            for wa1 in wa1_samples:
                u_dense = np.linspace(0, 1, n_dense + 1)
                u_mid = 0.5 * (u_dense[:-1] + u_dense[1:])
                du = u_dense[1] - u_dense[0]
                
                a_mid = 1.0 / (1.0 + z * u_mid)
                E_mid = _cpl_e(a_mid, wa0, wa1)
                integrand = 1.0 / E_mid
                true_val = np.sum(integrand) * du
                
                a_quad = 1.0 / (1.0 + z * nodes)
                E_quad = _cpl_e(a_quad, wa0, wa1)
                approx_val = np.sum(weights / E_quad)
                
                if true_val > 1e-12 and approx_val > 1e-12:
                    rel_err = abs(approx_val - true_val) / true_val
                    errors.append(rel_err)
    
    errors = np.array(errors)
    if len(errors) == 0:
        return 0.0
    
    # Log-sum-squared for smoothness
    return np.sum(np.log1p(errors**2))

def construct():
    # Comprehensive validation grid
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    wa0_samples = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    wa1_samples = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    # Use subset for optimization speed
    z_opt = z_samples[::2]
    wa0_opt = wa0_samples[::2]
    wa1_opt = wa1_samples[::2]
    
    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    
    # Invert to get params for our parameterization
    # nodes = sort(sigmoid(node_params))
    # We need node_params such that sort(sigmoid(node_params)) ≈ nodes_gl
    # Since sigmoid is monotonic, we can just use logit(nodes_gl)
    # But we need to sort them, so we need to assign params such that after sorting we get nodes_gl
    # Simple approach: use logit of the sorted nodes_gl
    nodes_gl_sorted = np.sort(nodes_gl)
    # Clip to avoid infinity
    nodes_gl_clipped = np.clip(nodes_gl_sorted, 1e-10, 1.0 - 1e-10)
    node_params_init = np.log(nodes_gl_clipped / (1.0 - nodes_gl_clipped))
    
    # Weight params: log(weights)
    weight_params_init = np.log(weights_gl)
    
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    # Phase 1: L-BFGS-B on smooth objective with reduced grid
    res1 = minimize(
        _objective_logsumsq,
        params_init,
        args=(z_opt, wa0_opt, wa1_opt, 500),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Phase 2: Powell on worst-case objective with full grid
    def _objective_worst_direct(params, z_samples, wa0_samples, wa1_samples, n_dense=1000):
        node_params = params[:8]
        weight_params = params[8:16]
        
        nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
        nodes = np.sort(nodes_raw)
        for i in range(1, 8):
            if nodes[i] - nodes[i-1] < 1e-12:
                nodes[i] = nodes[i-1] + 1e-12
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        weights = np.exp(weight_params)
        weights = weights / np.sum(weights)
        
        worst_err, _, _, _ = _compute_worst_error(nodes, weights, z_samples, wa0_samples, wa1_samples, n_dense)
        return worst_err
    
    res2 = minimize(
        _objective_worst_direct,
        res1.x,
        args=(z_samples, wa0_samples, wa1_samples, 1000),
        method='Powell',
        options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
    )
    
    # Extract final nodes and weights
    def _decode_params(params):
        node_params = params[:8]
        weight_params = params[8:16]
        
        nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
        nodes = np.sort(nodes_raw)
        for i in range(1, 8):
            if nodes[i] - nodes[i-1] < 1e-12:
                nodes[i] = nodes[i-1] + 1e-12
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        weights = np.exp(weight_params)
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    nodes1, weights1 = _decode_params(res1.x)
    nodes2, weights2 = _decode_params(res2.x)
    
    # Evaluate both on full grid
    worst1, _, _, _ = _compute_worst_error(nodes1, weights1, z_samples, wa0_samples, wa1_samples, 2000)
    worst2, _, _, _ = _compute_worst_error(nodes2, weights2, z_samples, wa0_samples, wa1_samples, 2000)
    
    if worst2 < worst1:
        nodes_opt, weights_opt = nodes2, weights2
    else:
        nodes_opt, weights_opt = nodes1, weights1
    
    # Final constraint enforcement
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes_opt[i] <= nodes_opt[i-1]:
            nodes_opt[i] = nodes_opt[i-1] + 1e-10
    
    # Ensure in (0,1)
    nodes_opt = np.clip(nodes_opt, 1e-10, 1.0 - 1e-10)
    
    # Ensure weights positive and sum to 1
    weights_opt = np.maximum(weights_opt, 1e-15)
    weights_opt = weights_opt / np.sum(weights_opt)
    
    return {"nodes": nodes_opt, "weights": weights_opt}
