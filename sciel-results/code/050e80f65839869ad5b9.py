"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre

def _cpl_e(a, wa0, wa1):
    """E(a) = H(a)/H0 for CPL model with Omega_m=0.3, Omega_L=0.7."""
    omega_m = 0.3
    omega_l = 0.7
    # E^2 = omega_m/a^3 + omega_l*(a/(1+a*wa1))^(3*(1+wa0))
    # Handle a=0 carefully, but nodes are in (0,1) so a is in (1/(1+z_max), 1)
    # For z up to 5, a_min is ~0.17, so no singularity at 0 for our range.
    term_m = omega_m / (a**3)
    base = a / (1.0 + a * wa1)
    # Ensure base is positive. If wa1 is very negative and a is large, base could be negative?
    # a is in (0,1]. 1+a*wa1 > 0 requires a < -1/wa1.
    # If wa1 = -1, a < 1. If wa1 = -0.5, a < 2.
    # Our a is in (0,1], so 1+a*wa1 is positive for wa1 > -1.
    # If wa1 <= -1, we might have issues. The task implies standard CPL ranges.
    # Let's assume wa1 > -1 or handle the power carefully.
    # Standard CPL: w(a) = wa0 + wa1*(1-a).
    # E^2 = Om/a^3 + Ol * exp(3 * integral_1^a (1+w(a'))/a' da')
    # For CPL, the integral is 3*(1+wa0)*ln(1+a*wa1)/wa1 + 3*wa1*(1-a) ... wait.
    # The formula in the seed is: term_l = omega_l * (a / (1.0 + a * wa1)) ** (3.0 * (1.0 + wa0))
    # Let's verify this formula.
    # H^2/H0^2 = Om/a^3 + Ol * (a/(1+a*wa1))^(3(1+wa0)) is a common approximation or specific form?
    # Actually, the exact solution for CPL w(a) = wa0 + wa1(1-a) is:
    # E^2(a) = Om/a^3 + Ol * exp( 3 * (1+wa0) * ln(1+a*wa1)/wa1 + 3*wa1*(1-a) )? No.
    # Let's stick to the seed's formula as it is the "published" one for this task context.
    # If base < 0 and exponent is not integer, we get NaN.
    # We will constrain wa1 in our test set to be > -1 to ensure validity.
    
    if base <= 0:
        # Fallback or return large value to penalize
        return 1e100
    term_l = omega_l * (base ** (3.0 * (1.0 + wa0)))
    return np.sqrt(term_m + term_l)

def _objective(params, z_samples, wa0_samples, wa1_samples, true_vals):
    """Compute sum of squared log-errors over a grid of z and cosmologies."""
    nodes = params[:8]
    log_weights = params[8:]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    # Check node ordering and bounds
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    
    total_error = 0.0
    count = 0
    
    # Pre-compute a and E for all nodes for all cosmologies?
    # nodes are fixed for this eval.
    # a_quad for a given z: 1/(1 + z*nodes)
    # E_quad depends on wa0, wa1.
    
    for i_z, z in enumerate(z_samples):
        a_quad = 1.0 / (1.0 + z * nodes)
        
        for i_wa0, wa0 in enumerate(wa0_samples):
            for i_wa1, wa1 in enumerate(wa1_samples):
                E_quad = _cpl_e(a_quad, wa0, wa1)
                
                # Check for invalid E
                if np.any(~np.isfinite(E_quad)) or np.any(E_quad <= 0):
                    return 1e10
                
                approx_val = np.sum(weights / E_quad)
                
                true_val = true_vals[i_z, i_wa0, i_wa1]
                
                if true_val > 1e-12 and approx_val > 1e-12:
                    rel_err = (approx_val - true_val) / true_val
                    # Use log1p(err^2) for robustness against outliers, or just err^2
                    # The directive mentions "smooth surrogate objective (e.g., sum of squared log-errors)"
                    # log(1+e^2) is smooth and bounded-ish.
                    total_error += np.log1p(rel_err**2)
                    count += 1
    
    return total_error / max(count, 1)

def _objective_max_rel_err(params, z_samples, wa0_samples, wa1_samples, true_vals):
    """Compute max absolute relative error."""
    nodes = params[:8]
    log_weights = params[8:]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
        
    max_err = 0.0
    
    for i_z, z in enumerate(z_samples):
        a_quad = 1.0 / (1.0 + z * nodes)
        
        for i_wa0, wa0 in enumerate(wa0_samples):
            for i_wa1, wa1 in enumerate(wa1_samples):
                E_quad = _cpl_e(a_quad, wa0, wa1)
                if np.any(~np.isfinite(E_quad)) or np.any(E_quad <= 0):
                    return 1e10
                
                approx_val = np.sum(weights / E_quad)
                true_val = true_vals[i_z, i_wa0, i_wa1]
                
                if true_val > 1e-12 and approx_val > 1e-12:
                    rel_err = abs((approx_val - true_val) / true_val)
                    if rel_err > max_err:
                        max_err = rel_err
    
    return max_err

def construct():
    # Define a grid of z values and CPL parameters to optimize over
    # z range: 0.1 to 3.0 to cover typical validation ranges
    z_samples = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    # CPL parameters: typical ranges
    wa0_samples = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    wa1_samples = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    
    # Pre-compute true integrals using dense midpoint rule
    # Integral I(z) = ∫_0^1 du / E(a(z*u)) where a(z*u) = 1/(1+z*u)
    n_dense = 1000
    u_dense = np.linspace(0, 1, n_dense + 1)
    u_mid = 0.5 * (u_dense[:-1] + u_dense[1:])
    du = u_dense[1] - u_dense[0]
    
    true_vals = np.zeros((len(z_samples), len(wa0_samples), len(wa1_samples)))
    
    for i_z, z in enumerate(z_samples):
        a_mid = 1.0 / (1.0 + z * u_mid)
        for i_wa0, wa0 in enumerate(wa0_samples):
            for i_wa1, wa1 in enumerate(wa1_samples):
                E_mid = _cpl_e(a_mid, wa0, wa1)
                integrand = 1.0 / E_mid
                true_vals[i_z, i_wa0, i_wa1] = np.sum(integrand) * du
    
    # Initial guess: Gauss-Legendre nodes and weights
    # Map GL nodes from [-1, 1] to [0, 1]
    x_gl, w_gl = roots_legendre(8)
    nodes_init = 0.5 * (x_gl + 1.0)
    weights_init = 0.5 * w_gl
    
    # Ensure weights are positive (they are for GL)
    # Sort nodes to be strictly increasing
    order = np.argsort(nodes_init)
    nodes_init = nodes_init[order]
    weights_init = weights_init[order]
    
    log_weights_init = np.log(weights_init)
    params_init = np.concatenate([nodes_init, log_weights_init])
    
    # Phase 1: L-BFGS-B on sum of squared log-errors
    # This provides a good local minimum for the smooth surrogate
    res1 = minimize(
        _objective,
        params_init,
        args=(z_samples, wa0_samples, wa1_samples, true_vals),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    # Phase 2: Powell refinement on max relative error
    # This directly targets the validation metric (worst-case error)
    # Powell is derivative-free and can handle the non-smooth max operation better
    res2 = minimize(
        _objective_max_rel_err,
        res1.x,
        args=(z_samples, wa0_samples, wa1_samples, true_vals),
        method='Powell',
        options={'maxiter': 200, 'ftol': 1e-14}
    )
    
    # Use the result with better worst-case error (or lower surrogate if max err is similar)
    # We evaluate both on the max error metric to decide
    err1 = _objective_max_rel_err(res1.x, z_samples, wa0_samples, wa1_samples, true_vals)
    err2 = _objective_max_rel_err(res2.x, z_samples, wa0_samples, wa1_samples, true_vals)
    
    if err2 < err1:
        params_opt = res2.x
    else:
        params_opt = res1.x
    
    # Extract optimized parameters
    nodes_opt = params_opt[:8]
    log_weights_opt = params_opt[8:]
    weights_opt = np.exp(log_weights_opt)
    weights_opt = weights_opt / np.sum(weights_opt)
    
    # Ensure nodes are strictly increasing and in (0, 1)
    order = np.argsort(nodes_opt)
    nodes_opt = nodes_opt[order]
    weights_opt = weights_opt[order]
    
    # Clip nodes to (0, 1) with small margin
    nodes_opt = np.clip(nodes_opt, 1e-8, 1.0 - 1e-8)
    
    # Ensure strict monotonicity
    for i in range(1, 8):
        if nodes_opt[i] <= nodes_opt[i-1]:
            nodes_opt[i] = nodes_opt[i-1] + 1e-8
    
    # Final normalization of weights
    weights_opt = weights_opt / np.sum(weights_opt)
    
    return {"nodes": nodes_opt, "weights": weights_opt}
# EVOLVE-BLOCK-END
