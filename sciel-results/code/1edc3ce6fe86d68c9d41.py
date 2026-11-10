"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad


def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless luminosity distance
    integral in spatially flat, time-varying dark-energy models.
    
    The integral is: dL/(c/H0) = (1+z) * ∫_0^z dz'/E(a(z'))
    where a = 1/(1+z') and E(a) = H(a)/H0.
    
    We use the change of variable x = z'/z, so the integral becomes:
    (1+z) * z * ∫_0^1 dx / E(a(x))
    where a(x) = 1/(1+z*x).
    
    The quadrature rule is: sum_i w_i / E(a_i) where a_i = 1/(1+z*nodes_i).
    
    We optimize nodes and weights to minimize worst-case relative error across a set
    of representative cosmologies (CPL parameterizations).
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    # Representative cosmologies for optimization (CPL parameters: w0, wa)
    # E(a) = (a^(-3(1+w0+wa)) * a^(-3wa))^(1/3) = a^(-1-w0-wa) * a^(-wa)
    # Actually for CPL: w(a) = w0 + wa*(1-a), so E(a) = a^(-3) * (a/(1-a))^(3wa/2) ... 
    # Let me use the standard form: E(a) = (Omega_m a^{-3} + Omega_r a^{-4} + Omega_L a^{-3(1+w0+wa)} * (a/(1-a))^{-3wa})^(1/2)
    # For simplicity, let's use a set of (w0, wa) pairs and assume Omega_m=0.3, Omega_r=0, Omega_L=0.7
    # E(a) = sqrt(0.3*a^{-3} + 0.7*a^{-3(1+w0+wa)}*(a/(1-a))^{-3wa})
    
    # Actually, let's use a simpler but representative set. The key is that the
    # integrand 1/E(a(x)) varies in shape depending on cosmology.
    
    # Let's define several test cosmologies with different (w0, wa)
    # and compute the reference integral using adaptive quadrature.
    
    Omega_m = 0.3
    Omega_L = 0.7
    Omega_r = 0.0
    
    # Test cosmologies: (w0, wa)
    test_cosmos = [
        (-1.0, 0.0),    # LambdaCDM
        (-1.0, 0.5),    # CPL with positive wa
        (-1.0, -0.5),   # CPL with negative wa
        (-0.5, 0.0),    # w0 > -1
        (-1.5, 0.0),    # w0 < -1
        (-1.0, 1.0),    # strong CPL
        (-1.0, -1.0),   # strong negative CPL
        (-0.5, 0.5),    # mixed
        (-1.5, 0.5),    # mixed
        (-0.5, -0.5),   # mixed
    ]
    
    def E_of_a(a, w0, wa):
        """H(a)/H0 for flat universe with CPL dark energy."""
        # E(a)^2 = Omega_m/a^3 + Omega_r/a^4 + Omega_L * a^{-3(1+w0+wa)} * (a/(1-a))^{-3wa}
        # Simplify: the DE term is Omega_L * a^{-3(1+w0+wa)} * (a/(1-a))^{-3wa}
        # = Omega_L * a^{-3-3w0-3wa} * a^{3wa} * (1-a)^{3wa}
        # = Omega_L * a^{-3-3w0} * (1-a)^{3wa}
        # Wait, let me re-derive.
        # rho_DE(a) = rho_DE0 * a^{-3(1+w(a))} where w(a) = w0 + wa*(1-a)
        # So rho_DE(a)/rho_DE0 = a^{-3(1+w0+wa*(1-a))}
        # = a^{-3-3w0-3wa+3wa*a}
        # E(a)^2 = Omega_m * a^{-3} + Omega_L * a^{-3-3w0-3wa+3wa*a}
        
        a_safe = np.clip(a, 1e-10, 1.0 - 1e-10)
        exp_m = -3.0
        exp_de = -3.0 - 3.0*w0 - 3.0*wa + 3.0*wa*a_safe
        E2 = Omega_m * np.exp(exp_m * np.log(a_safe)) + Omega_L * np.exp(exp_de * np.log(a_safe))
        E2 = np.maximum(E2, 1e-30)
        return np.sqrt(E2)
    
    def reference_integral(z, w0, wa):
        """Compute the reference value of z * ∫_0^1 dx/E(a(x)) using adaptive quad."""
        a_x = 1.0 / (1.0 + z * 0.5)  # midpoint for quad
        def integrand(x):
            a = 1.0 / (1.0 + z * x)
            return 1.0 / E_of_a(a, w0, wa)
        
        val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
        return z * val
    
    def quadrature_value(nodes, weights, z, w0, wa):
        """Compute the quadrature approximation: z * sum_i w_i / E(a_i)"""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return z * np.sum(weights / E_vals)
    
    def relative_error(nodes, weights, z, w0, wa):
        """Relative error in percent."""
        ref = reference_integral(z, w0, wa)
        approx = quadrature_value(nodes, weights, z, w0, wa)
        if abs(ref) < 1e-15:
            return 0.0
        return abs(approx - ref) / abs(ref) * 100.0
    
    def worst_case_error(nodes, weights):
        """Worst relative error across test cosmologies and several redshifts."""
        max_err = 0.0
        for w0, wa in test_cosmos:
            for z in [0.1, 0.5, 1.0, 2.0, 5.0]:
                err = relative_error(nodes, weights, z, w0, wa)
                max_err = max(max_err, err)
        return max_err
    
    # Parameterize: nodes in [0,1] strictly increasing, weights positive summing to 1
    # Use log-weights for positivity, and a parameterization for nodes.
    
    def params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights."""
        # First 8 params for nodes (use tanh to map to (0,1)), then 7 for log-weights
        node_params = params[:8]
        weight_params = params[8:15]  # 7 params for 8 weights (last determined by sum)
        
        # Map node_params to (0,1) using tanh, then sort and add small offsets
        nodes_raw = 0.5 * (1.0 + np.tanh(node_params))
        # Ensure strictly increasing by sorting and adding small increments
        nodes_sorted = np.sort(nodes_raw)
        # Add small increments to ensure strict increase
        nodes = nodes_sorted.copy()
        min_gap = 1e-8
        for i in range(1, 8):
            if nodes[i] - nodes[i-1] < min_gap:
                nodes[i] = nodes[i-1] + min_gap
        # Clamp to [epsilon, 1-epsilon]
        eps = 1e-6
        nodes[0] = max(nodes[0], eps)
        nodes[-1] = min(nodes[-1], 1.0 - eps)
        
        # Weights: use softmax-like to ensure positivity and sum to 1
        # Use 7 free weights, 8th determined
        w_raw = np.exp(weight_params)
        w8 = 1.0  # free parameter for 8th weight
        # Actually, let's use a different approach: 8 log-weights, normalize
        # But we only have 7 params. Let's use: weights_i = exp(logw_i) for i=1..7, 
        # and weight_8 = 1 - sum(exp(logw_i)), ensuring all positive.
        # Better: use softmax on 8 values where one is fixed.
        
        # Let's use: w_i = exp(p_i) / (sum(exp(p_j)) + exp(p_8)) for i=1..7
        # and w_8 = exp(p_8) / (sum(exp(p_j)) + exp(p_8))
        # where p_8 = 0 (fixed). This ensures positivity and sum=1.
        
        p = np.zeros(8)
        p[:7] = weight_params
        # p[7] = 0.0 (fixed)
        
        # Softmax
        p_max = np.max(p)
        exp_p = np.exp(p - p_max)
        weights = exp_p / np.sum(exp_p)
        
        return nodes, weights
    
    def objective(params):
        """Objective function for optimization."""
        nodes, weights = params_to_nodes_weights(params)
        
        # Check constraints
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e6
        
        return worst_case_error(nodes, weights)
    
    # Initial guess: midpoint rule
    n = 8
    initial_nodes = (np.arange(n) + 0.5) / n
    initial_weights = np.full(n, 1.0 / n)
    
    # Convert to params
    # For nodes: inverse tanh
    node_params_init = np.arctanh(2 * initial_nodes - 1)
    # For weights: log of weights, with 8th weight's log set to 0
    # weights_i = exp(p_i) / sum(exp(p_j))
    # If we want weights_i = 1/8, then p_i = 0 for all i
    weight_params_init = np.zeros(7)
    
    x0 = np.concatenate([node_params_init, weight_params_init])
    
    # Try multiple optimization methods
    best_params = x0.copy()
    best_obj = objective(x0)
    
    # Method 1: Nelder-Mead
    try:
        result1 = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if result1.fun < best_obj:
            best_obj = result1.fun
            best_params = result1.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        nodes, weights = params_to_nodes_weights(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Method 2: Powell
    try:
        result2 = minimize(objective, best_params, method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
        if result2.fun < best_obj:
            best_obj = result2.fun
            best_params = result2.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        nodes, weights = params_to_nodes_weights(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Method 3: L-BFGS-B with finite differences
    try:
        result3 = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        if result3.fun < best_obj:
            best_obj = result3.fun
            best_params = result3.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        nodes, weights = params_to_nodes_weights(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Final refinement with Nelder-Mead from best point
    try:
        result4 = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if result4.fun < best_obj:
            best_obj = result4.fun
            best_params = result4.x
    except Exception:
        pass
    
    nodes, weights = params_to_nodes_weights(best_params)
    
    # Final validation
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0) and 
            np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-12):
        # Fallback to midpoint rule
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
