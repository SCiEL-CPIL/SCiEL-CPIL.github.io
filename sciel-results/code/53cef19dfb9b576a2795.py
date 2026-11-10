import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Gauss-Legendre 8-point nodes and weights on [-1, 1] as initial guess
    t_gl = np.array([
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346425004484,
         0.1834346425004484,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    ])
    
    nodes_gl = 0.5 * t_gl + 0.5
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Best solution so far
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_obj = np.inf
    
    # Define a dense grid of CPL models and redshifts for robust optimization
    # w0 in [-1.5, 0.5], wa in [-1.5, 1.5]
    # z in [0.01, 10]
    
    # We will sample a large number of points to approximate the validation distribution
    # Using a stratified sampling approach for better coverage
    np.random.seed(42)
    n_samples = 200
    w0_samples = np.random.uniform(-1.5, 0.5, n_samples)
    wa_samples = np.random.uniform(-1.5, 1.5, n_samples)
    z_samples = np.logspace(-2, 1, n_samples) # 0.01 to 10
    
    # Add specific "edge" cases that are often hard
    w0_edges = np.array([-1.0, -1.5, 0.0, 0.5])
    wa_edges = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    z_edges = np.array([0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    # Combine all test cases
    w0_all = np.concatenate([w0_samples, np.repeat(w0_edges, len(wa_edges)*len(z_edges))])
    wa_all = np.concatenate([wa_samples, np.repeat(wa_edges, len(z_edges))])
    # For edges, we want a grid product
    w0_grid, wa_grid, z_grid = np.meshgrid(w0_edges, wa_edges, z_edges, indexing='ij')
    w0_all = np.concatenate([w0_samples, w0_grid.flatten()])
    wa_all = np.concatenate([wa_samples, wa_grid.flatten()])
    z_all = np.concatenate([z_samples, z_grid.flatten()])
    
    # Pre-compute dense integration values for these test cases?
    # No, we compute on the fly but we can cache if needed. 
    # Given n_samples=200 + edges, total cases ~ 200 + 4*5*7 = 340.
    # Each case requires a dense integration (e.g., 1000 points) for ground truth.
    # 340 * 1000 = 340k operations per objective evaluation.
    # This is fast enough for L-BFGS-B.
    
    def _E(a, w0, wa):
        """H(a)/H0 for CPL model. a = 1/(1+z)"""
        Om_m = 0.3
        Om_La = 1.0 - Om_m
        # Exponent for DE: 3*(1+w0+wa*(1-a))
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        term_m = Om_m * a**(-3.0)
        term_de = Om_La * a**(exponent_de)
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))
    
    def _true_integral_vectorized(w0, wa, z):
        """Compute true integral using dense quadrature for vectorized inputs"""
        # To speed up, we compute the dense integral for all test cases at once?
        # Or just loop over test cases. Looping is safer for memory.
        
        # We will store the "true" integrals in a global cache to avoid recomputing
        # if the test set is fixed.
        pass # Placeholder, we'll compute in the objective if not cached.

    # Cache for true integrals
    true_int_cache = {}
    
    def _get_true_integral(w0, wa, z):
        key = (w0, wa, z)
        if key not in true_int_cache:
            # Dense integration
            u = np.linspace(0, 1, 1001)
            a = 1.0 / (1.0 + z * u)
            E_vals = _E(a, w0, wa)
            integrand = 1.0 / E_vals
            true_int = np.trapz(integrand, u)
            true_val = (1.0 + z) * z * true_int
            true_int_cache[key] = true_val
        return true_int_cache[key]

    def _quadrature_value(nodes, weights, w0, wa, z):
        """Compute quadrature approximation"""
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    def _objective_minmax(x):
        """Objective: minimize worst-case relative error"""
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        
        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)
        
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
            
        max_rel_err = 0.0
        for i in range(len(w0_all)):
            w0 = w0_all[i]
            wa = wa_all[i]
            z = z_all[i]
            
            true_val = _get_true_integral(w0, wa, z)
            quad_val = _quadrature_value(nodes, weights, w0, wa, z)
            
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        return max_rel_err

    def _objective_ls(x):
        """Objective: minimize sum of squared log-errors (smooth surrogate)"""
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        
        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)
        
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
            
        sum_sq_err = 0.0
        for i in range(len(w0_all)):
            w0 = w0_all[i]
            wa = wa_all[i]
            z = z_all[i]
            
            true_val = _get_true_integral(w0, wa, z)
            quad_val = _quadrature_value(nodes, weights, w0, wa, z)
            
            if true_val > 1e-10 and quad_val > 1e-10:
                rel_err = (quad_val - true_val) / true_val
                sum_sq_err += rel_err**2
        return sum_sq_err

    def _encode(nodes, weights):
        """Encode nodes and weights into parameter vector"""
        spacings = np.diff(nodes)
        # Ensure positive for log
        spacings = np.maximum(spacings, 1e-15)
        log_spacings = np.log(spacings)
        
        weights = np.maximum(weights, 1e-15)
        log_weights = np.log(weights)
        return np.concatenate([log_spacings, log_weights])
    
    def _decode(x):
        """Decode parameter vector to nodes and weights"""
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        
        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)
        
        return nodes, weights

    x0 = _encode(best_nodes, best_weights)
    
    # Step 1: Global Search with Differential Evolution on LS objective
    # DE is robust to non-convexity and finds good basins.
    if time.time() - start_time < deadline - 20:
        try:
            bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
            de_res = differential_evolution(_objective_ls, bounds, 
                                           maxiter=30, popsize=15, 
                                           tol=1e-4, seed=42, 
                                           polish=False,
                                           init='latinhypercube')
            if de_res.fun < 1e10: # Check validity
                best_nodes, best_weights = _decode(de_res.x)
                # Update best if valid
                if np.all(best_nodes > 1e-15) and np.all(best_nodes < 1 - 1e-15) and \
                   np.all(np.diff(best_nodes) > 1e-15) and np.all(best_weights > 0):
                    # Evaluate min-max to see if it's actually good
                    current_mm = _objective_minmax(de_res.x)
                    if current_mm < best_obj:
                        best_obj = current_mm
        except Exception:
            pass
    
    # Step 2: Local Refinement with L-BFGS-B on LS objective
    if time.time() - start_time < deadline - 10:
        try:
            x_current = _encode(best_nodes, best_weights)
            lbfgs_res = minimize(_objective_ls, x_current, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if lbfgs_res.fun < 1e10:
                candidate_nodes, candidate_weights = _decode(lbfgs_res.x)
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    # Check min-max improvement
                    current_mm = _objective_minmax(lbfgs_res.x)
                    if current_mm < best_obj:
                        best_obj = current_mm
                        best_nodes = candidate_nodes
                        best_weights = candidate_weights
        except Exception:
            pass
            
    # Step 3: Final Polish with Nelder-Mead on Min-Max objective
    # This targets the specific metric we need to minimize.
    if time.time() - start_time < deadline - 2:
        try:
            x_current = _encode(best_nodes, best_weights)
            nm_res = minimize(_objective_minmax, x_current, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if nm_res.fun < best_obj:
                candidate_nodes, candidate_weights = _decode(nm_res.x)
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
                    best_obj = nm_res.fun
        except Exception:
            pass
    
    # Final validation and normalization
    best_weights = best_weights / np.sum(best_weights)
    
    # Verify all constraints
    if not (len(best_nodes) == 8 and len(best_weights) == 8 and
            np.all(best_nodes > 0) and np.all(best_nodes < 1) and
            np.all(np.diff(best_nodes) > 0) and
            np.all(best_weights > 0) and
            abs(np.sum(best_weights) - 1.0) < 1e-12):
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
