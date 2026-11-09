"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    a_safe = np.maximum(a, 1e-10)
    # Standard CPL: w(z) = w0 + wa * z / (1+z) = w0 + wa * (1-a)/a
    w_eff = w0 + wa * (1.0 - a_safe) / a_safe
    exponent_de = -3.0 * (1.0 + w_eff)
    
    h2 = om_m * a_safe**(-3) + om_r * a_safe**(-4) + om_k * a_safe**0 + \
         (1 - om_m - om_r - om_k) * a_safe**exponent_de
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand_val(x, z, w0, wa, om_m, om_r, om_k):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)

def _true_dL_integral(z, w0, wa, om_m, om_r, om_k, n=2000):
    """Compute the true integral part using high-resolution Simpson's rule."""
    x = np.linspace(0, 1, n)
    if n % 2 == 0:
        n += 1
        x = np.linspace(0, 1, n)
    
    y = _integrand_val(x, z, w0, wa, om_m, om_r, om_k)
    
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return integral

def _generate_extremal_models():
    """
    Generate a set of cosmological models that likely bound the behavior of the integrand.
    We focus on extremes of curvature and slope of 1/E(a).
    """
    models = []
    # Extremes of w0 and wa
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    om_m_vals = [0.2, 0.3, 0.4]
    om_r_vals = [0.0, 0.1]
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for om_m in om_m_vals:
                for om_r in om_r_vals:
                    om_de = 1.0 - om_m - om_r
                    if om_de <= 0:
                        continue
                    models.append((w0, wa, om_m, om_r, 0.0))
                    
    # Add a few random samples to ensure coverage
    np.random.seed(42)
    for _ in range(50):
        w0 = np.random.uniform(-1.5, 0.5)
        wa = np.random.uniform(-1.0, 1.0)
        om_m = np.random.uniform(0.2, 0.4)
        om_r = np.random.uniform(0.0, 0.1)
        om_de = 1.0 - om_m - om_r
        if om_de > 0:
            models.append((w0, wa, om_m, om_r, 0.0))
            
    return models

def _compute_moments_for_init(models, z_vals, n_nodes=8):
    """
    Use moment matching on the most 'complex' integrand to initialize nodes/weights.
    We pick a model with high curvature (e.g., extreme wa) and match its moments.
    """
    # Pick a representative 'hard' model
    # High |wa| usually means more curvature in 1/E(a)
    hard_model = models[0] # Default
    max_curvature = 0
    for m in models:
        # Estimate curvature by sampling
        z_test = 1.0
        x_test = np.linspace(0.01, 0.99, 100)
        y_test = _integrand_val(x_test, z_test, *m)
        # Approximate second derivative magnitude
        if len(y_test) > 2:
            d2y = np.diff(y_test, 2)
            curvature_metric = np.max(np.abs(d2y))
            if curvature_metric > max_curvature:
                max_curvature = curvature_metric
                hard_model = m
                
    w0, wa, om_m, om_r, om_k = hard_model
    z_rep = 1.0 # Representative redshift
    
    # We want nodes x_i and weights w_i such that:
    # sum(w_i * f(x_i)^k) ~ integral(f(x)^k dx) for k=0..n-1
    # This is a non-linear system. We can approximate by finding nodes that
    # match the distribution of the function values.
    
    # Alternative: Use Gauss-Legendre as a strong baseline, then perturb.
    # But let's try to match the first few moments of the specific function f(x).
    x_fine = np.linspace(0, 1, 1000)
    f_fine = _integrand_val(x_fine, z_rep, w0, wa, om_m, om_r, om_k)
    
    # Moments of f(x)
    moments_true = np.zeros(n_nodes)
    for k in range(n_nodes):
        moments_true[k] = np.trapz(f_fine**k, x_fine)
        
    # We need to find x_i, w_i such that sum(w_i * f(x_i)^k) = moments_true[k]
    # This is hard to solve directly. 
    # Instead, we can use the nodes from Gauss-Legendre but adjust weights to match moments of f(x).
    # Or, simpler: Use a set of nodes that are "clustered" where f(x) changes rapidly.
    
    # For now, let's use a heuristic initialization:
    # Nodes are spread out, but we can refine them.
    # A good initialization is the Gauss-Legendre nodes, as they are optimal for polynomials.
    # Since 1/E(a) is smooth, GL is a good start.
    
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)
    nodes = (x_gl + 1.0) / 2.0
    weights = w_gl / 2.0
    
    return nodes, weights

def _objective_max_err(params, models, z_vals, true_integrals):
    """
    Compute the maximum relative error across all models and z_vals.
    params: 16 floats [8 nodes, 8 weights]
    """
    nodes = params[:8]
    weights = params[8:]
    
    # Check validity
    if np.any(nodes < 0) or np.any(nodes > 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        # Penalize non-normalized weights heavily, but don't fail immediately
        # Actually, for minimax, we should normalize internally or penalize.
        # Let's normalize internally to keep the search space smooth.
        weights = weights / np.sum(weights)

    max_err = 0.0
    n_models = len(models)
    n_z = len(z_vals)
    
    for i_m in range(n_models):
        w0, wa, om_m, om_r, om_k = models[i_m]
        for i_z in range(n_z):
            z = z_vals[i_z]
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
            
            # The quadrature rule is: sum(weights[i] / E(a_i))
            # But wait, the formula is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
            # The "true" integral we computed in _true_dL_integral is the integral part:
            # int_0^z dz' / E(1/(1+z'))
            # Let's check the variable change in _true_dL_integral.
            # _true_dL_integral uses x in [0,1] where a = 1/(1+z*x).
            # So the integral is int_0^1 dx / E(1/(1+z*x)).
            # The quadrature approximates this integral.
            
            quad_val = np.sum(weights / E_vals)
            true_val = true_integrals[i_m, i_z]
            
            if true_val < 1e-10:
                continue
                
            err = abs(quad_val - true_val) / true_val
            if err > max_err:
                max_err = err
                if max_err > 1.0: # Early exit
                    return max_err
                    
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Define a dense, diverse set of "training" cosmologies.
    models = _generate_extremal_models()
    # Limit to a manageable number for optimization speed
    if len(models) > 80:
        indices = np.random.choice(len(models), 80, replace=False)
        models = [models[i] for i in indices]
        
    z_vals = np.logspace(-1, 2.5, 15) # 0.1 to ~316
    
    # Pre-compute true integrals
    true_integrals = np.zeros((len(models), len(z_vals)))
    for i_m, m in enumerate(models):
        w0, wa, om_m, om_r, om_k = m
        for i_z, z in enumerate(z_vals):
            true_integrals[i_m, i_z] = _true_dL_integral(z, w0, wa, om_m, om_r, om_k)
            
    # 2. Initialization
    nodes_init, weights_init = _compute_moments_for_init(models, z_vals)
    params_init = np.concatenate([nodes_init, weights_init])
    
    # 3. Optimization Strategy: Alternating Minimax
    # We will alternate between:
    # A) Optimizing nodes (non-linear) with weights fixed.
    # B) Optimizing weights (linear-ish) with nodes fixed.
    
    current_params = params_init.copy()
    best_params = params_init.copy()
    best_err = _objective_max_err(best_params, models, z_vals, true_integrals)
    
    max_iter = 50
    for i_iter in range(max_iter):
        if time.time() > deadline - 15:
            break
            
        nodes = current_params[:8]
        weights = current_params[8:]
        
        # Step A: Optimize Nodes
        # We use Nelder-Mead or Powell on the 8-node vector.
        # Objective: max error with current weights.
        
        def obj_nodes(n_params):
            n_nodes = n_params
            # Ensure strict ordering
            if np.any(np.diff(n_nodes) <= 0):
                return 1e5
            if np.any(n_nodes < 0) or np.any(n_nodes > 1):
                return 1e5
            temp_params = np.concatenate([n_nodes, weights])
            return _objective_max_err(temp_params, models, z_vals, true_integrals)
            
        try:
            res_nodes = minimize(
                obj_nodes, 
                nodes, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nodes.fun < best_err:
                current_params[:8] = res_nodes.x
                best_params = current_params.copy()
                best_err = res_nodes.fun
        except Exception:
            pass
            
        if time.time() > deadline - 15:
            break
            
        # Step B: Optimize Weights
        # We want to minimize max |sum(w_i/E_i) - T|.
        # This is a linear programming problem if we fix the max error M:
        # Minimize M
        # Subject to:
        # -M <= sum(w_i/E_i) - T <= M for all (m, z)
        # sum(w_i) = 1
        # w_i > 0
        
        # However, linprog minimizes c^T x. We can set c=0 and minimize M by adding M to the variables.
        # Variables: w_1...w_8, M
        # Constraints:
        # For each (m, z):
        # sum(w_i/E_i) - T <= M
        # -sum(w_i/E_i) + T <= M
        
        # Let's build the constraint matrix.
        n_constraints = len(models) * len(z_vals)
        # A_ub shape: (n_constraints * 2, 9)
        # b_ub shape: (n_constraints * 2,)
        
        A_ub = np.zeros((n_constraints * 2, 9))
        b_ub = np.zeros(n_constraints * 2)
        
        idx = 0
        for i_m in range(len(models)):
            w0, wa, om_m, om_r, om_k = models[i_m]
            for i_z in range(len(z_vals)):
                z = z_vals[i_z]
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
                inv_E = 1.0 / E_vals
                true_val = true_integrals[i_m, i_z]
                
                # Constraint 1: sum(w_i * inv_E_i) - true_val <= M
                # sum(w_i * inv_E_i) - M <= true_val
                A_ub[idx, :8] = inv_E
                A_ub[idx, 8] = -1.0
                b_ub[idx] = true_val
                idx += 1
                
                # Constraint 2: -(sum(w_i * inv_E_i) - true_val) <= M
                # -sum(w_i * inv_E_i) + M <= -true_val
                A_ub[idx, :8] = -inv_E
                A_ub[idx, 8] = 1.0
                b_ub[idx] = -true_val
                idx += 1
                
        # Equality constraint: sum(w_i) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, M >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        c = np.zeros(9)
        c[8] = 1.0 # Minimize M
        
        try:
            res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res_lp.success:
                new_weights = res_lp.x[:8]
                new_M = res_lp.x[8]
                
                # Check if this solution is better
                # The LP minimizes M, which is the max absolute error.
                # We want max relative error. M is absolute error.
                # Since true_val varies, minimizing absolute M is a good proxy, 
                # but might not be optimal for relative error.
                # However, it's a strong step.
                
                temp_params = np.concatenate([nodes, new_weights])
                err_rel = _objective_max_err(temp_params, models, z_vals, true_integrals)
                
                if err_rel < best_err:
                    current_params[8:] = new_weights
                    best_params = current_params.copy()
                    best_err = err_rel
        except Exception:
            pass
            
    # Final Polish: Run a short DE on the best params found so far to escape any local minima
    if time.time() < deadline - 5:
        # Use DE with a small population around the best solution
        # We need to define bounds around the current best
        bounds_de = []
        for i in range(8):
            bounds_de.append((max(0.01, best_params[i]-0.1), min(0.99, best_params[i]+0.1)))
        for i in range(8):
            bounds_de.append((max(0.01, best_params[8+i]-0.1), min(0.5, best_params[8+i]+0.1)))
            
        def obj_de(params):
            nodes = params[:8]
            weights = params[8:]
            if np.any(np.diff(nodes) <= 0):
                return 1e5
            if np.any(nodes < 0) or np.any(nodes > 1):
                return 1e5
            if np.any(weights <= 0):
                return 1e5
            weights_norm = weights / np.sum(weights)
            temp_params = np.concatenate([nodes, weights_norm])
            return _objective_max_err(temp_params, models, z_vals, true_integrals)
            
        try:
            res_de = differential_evolution(
                obj_de, 
                bounds_de, 
                maxiter=20, 
                popsize=5, 
                seed=42,
                polish=False # Skip polish to save time
            )
            if res_de.fun < best_err:
                best_params = res_de.x
                best_err = res_de.fun
        except Exception:
            pass
            
    # Final Validation and Normalization
    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Ensure strict ordering
    if not np.all(np.diff(nodes) > 0):
        # Fallback to Gauss-Legendre
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
    else:
        weights = weights / np.sum(weights)
        
    # One final check: if the error is terrible, fallback to GL
    test_models = [
        (0.0, 0.0, 0.3, 0.05, 0.0),
        (-1.0, 0.0, 0.3, 0.05, 0.0),
        (-0.5, 0.5, 0.3, 0.05, 0.0)
    ]
    test_z = [0.5, 1.0, 2.0]
    
    max_err = 0
    for m in test_models:
        for z in test_z:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, m[0], m[1], m[2], m[3], m[4])
            quad_val = np.sum(weights / E_vals)
            true_val = _true_dL_integral(z, m[0], m[1], m[2], m[3], m[4])
            if true_val > 0:
                err = abs(quad_val - true_val) / true_val
                if err > max_err:
                    max_err = err
                    
    if max_err > 0.1: # 10% error is bad
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0

    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
