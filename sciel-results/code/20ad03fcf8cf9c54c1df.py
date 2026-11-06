"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _worst_error(nodes, weights, z_samples, models):
    """Compute worst-case relative error across all (z, model) pairs."""
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def _soft_worst_error(nodes, weights, z_samples, models, temperature=1.0):
    """Smooth approximation of worst error using log-sum-exp."""
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            errors.append(err)
    errors = np.array(errors)
    if len(errors) == 0:
        return 0.0
    max_err = np.max(errors)
    if max_err < 1e-15:
        return 0.0
    # Soft max approximation
    scaled = errors / max(temperature, 1e-10)
    log_sum_exp = np.log(np.sum(np.exp(scaled - np.max(scaled)))) + np.max(scaled)
    return max_err * np.exp(log_sum_exp - np.max(scaled))

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline (15s margin for safety)
    
    n = 8
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
        (-0.1, 0.2, -1.1),      # Another phantom
        (0.15, -0.4, -0.95),    # More varied
    ]
    
    # Z samples for evaluation - include a range of redshifts
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Start with Gauss-Legendre nodes on [0,1] as initial guess
    gl_nodes, _ = np.polynomial.legendre.leggauss(n)
    best_nodes = 0.5 * (gl_nodes + 1.0)  # Map from [-1,1] to [0,1]
    best_weights = np.full(n, 1.0 / n)
    
    # Initial evaluation
    best_score = _worst_error(best_nodes, best_weights, z_samples, models)
    
    # Strategy: Use differential evolution for global exploration, then local refinement
    
    def _params_to_nodes_weights(params):
        """Convert parameter vector to valid nodes and weights."""
        # Nodes: use sigmoid to map to (0, 1)
        raw_nodes = params[:n]
        nodes = 1.0 / (1.0 + np.exp(-raw_nodes * 10))  # Scaled sigmoid
        nodes = np.sort(nodes)
        
        # Weights: use softmax to ensure positive and sum to 1
        raw_weights = params[n:]
        weights = np.exp(raw_weights - np.max(raw_weights))
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    def _objective(params):
        """Objective function for optimization."""
        nodes, weights = _params_to_nodes_weights(params)
        
        # Check for degenerate nodes
        if np.any(np.diff(nodes) < 1e-8):
            return 1e6
        
        return _worst_error(nodes, weights, z_samples, models)
    
    # Phase 1: Differential Evolution for global search
    # Bounds for raw parameters
    bounds = [(-5, 5)] * n + [(-5, 5)] * n
    
    # Run differential evolution with limited budget
    de_start = time.time()
    de_deadline = start_time + 40.0  # 40s for DE
    
    try:
        de_result = differential_evolution(
            _objective, bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 0.8),
            recombination=0.9,
            polish=False,
            update_interval=1,
            callback=lambda xk, convergence: None
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_nodes, best_weights = _params_to_nodes_weights(de_result.x)
            
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        # Not enough time for refinement
        best_nodes = np.sort(best_nodes)
        best_weights = np.maximum(best_weights, 1e-15)
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Phase 2: Local refinement using Nelder-Mead or SLSQP
    # Convert current best to raw parameter space
    raw_nodes_best = np.log(best_nodes / (1 - best_nodes)) / 10.0  # Inverse sigmoid
    raw_weights_best = np.log(best_weights)  # Inverse softmax (approximately)
    
    x0 = np.concatenate([raw_nodes_best, raw_weights_best])
    
    # Try multiple local optimization methods
    local_methods = [
        ('Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}),
        ('Powell', {'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10}),
    ]
    
    for method_name, options in local_methods:
        if time.time() > deadline - 10:
            break
        
        try:
            local_result = minimize(
                _objective, x0,
                method=method_name,
                options=options
            )
            
            if local_result.fun < best_score:
                best_score = local_result.fun
                best_nodes, best_weights = _params_to_nodes_weights(local_result.x)
                x0 = local_result.x  # Update starting point for next method
                
        except Exception:
            continue
    
    # Phase 3: Fine-tune with SLSQP if time permits
    if time.time() < deadline - 5:
        try:
            # Direct optimization in node/weight space with constraints
            def obj_direct(params):
                nodes = params[:n]
                weights = params[n:]
                
                if np.any(nodes <= 0) or np.any(nodes >= 1):
                    return 1e6
                if np.any(np.diff(nodes) <= 1e-10):
                    return 1e6
                if np.any(weights <= 0):
                    return 1e6
                if abs(np.sum(weights) - 1.0) > 1e-8:
                    return 1e6
                
                return _worst_error(nodes, weights, z_samples, models)
            
            def constraint_nodes_increasing(params):
                nodes = params[:n]
                return np.diff(nodes) - 1e-6
            
            def constraint_weights_sum(params):
                weights = params[n:]
                return np.sum(weights) - 1.0
            
            bounds_direct = [(0.001, 0.999)] * n + [(1e-6, 0.5)] * n
            x0_direct = np.concatenate([best_nodes, best_weights])
            
            slsqp_result = minimize(
                obj_direct, x0_direct,
                method='SLSQP',
                bounds=bounds_direct,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_nodes_increasing},
                    {'type': 'eq', 'fun': constraint_weights_sum}
                ],
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if slsqp_result.fun < best_score:
                best_score = slsqp_result.fun
                best_nodes = slsqp_result.x[:n]
                best_weights = slsqp_result.x[n:]
                
        except Exception:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    final_errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            final_errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
