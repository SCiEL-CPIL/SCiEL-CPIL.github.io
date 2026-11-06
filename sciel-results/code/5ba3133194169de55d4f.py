"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
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

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline (10s margin)
    
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
    ]
    
    # Z samples for evaluation - include a range of redshifts
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    # Start with Gauss-Legendre nodes on [0,1] as initial guess
    gl_nodes, _ = np.polynomial.legendre.leggauss(n)
    best_nodes = 0.5 * (gl_nodes + 1.0)  # Map from [-1,1] to [0,1]
    best_weights = np.full(n, 1.0 / n)
    
    # Initial evaluation
    best_score = _worst_error(best_nodes, best_weights, z_samples, models)
    
    # Joint optimization of nodes and weights using SLSQP
    # Parameters: [nodes_0, ..., nodes_7, weights_0, ..., weights_7]
    # Constraints: nodes strictly increasing in (0,1), weights positive and sum to 1
    
    def objective(params):
        nodes = params[:n]
        weights = params[n:]
        
        # Check validity
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 1e-10):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e6
        
        return _worst_error(nodes, weights, z_samples, models)
    
    # Constraints for SLSQP
    def constraint_nodes_increasing(params):
        nodes = params[:n]
        return np.diff(nodes) - 1e-6  # Ensure strictly increasing with margin
    
    def constraint_weights_sum(params):
        weights = params[n:]
        return np.sum(weights) - 1.0
    
    # Bounds: nodes in (0.001, 0.999), weights in (1e-6, 0.5)
    bounds = [(0.001, 0.999)] * n + [(1e-6, 0.5)] * n
    
    # Initial guess
    x0 = np.concatenate([best_nodes, best_weights])
    
    # Try multiple initial points
    initial_guesses = [x0]
    
    # Add some perturbed initial guesses
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        nodes_pert = best_nodes + rng.normal(0, 0.02, n)
        nodes_pert = np.clip(nodes_pert, 0.001, 0.999)
        nodes_pert = np.sort(nodes_pert)
        weights_pert = best_weights + rng.normal(0, 0.01, n)
        weights_pert = np.maximum(weights_pert, 1e-6)
        weights_pert = weights_pert / np.sum(weights_pert)
        initial_guesses.append(np.concatenate([nodes_pert, weights_pert]))
    
    best_params = x0
    best_obj = best_score
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                objective, guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_nodes_increasing},
                    {'type': 'eq', 'fun': constraint_weights_sum}
                ],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            continue
    
    best_nodes = best_params[:n]
    best_weights = best_params[n:]
    
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
