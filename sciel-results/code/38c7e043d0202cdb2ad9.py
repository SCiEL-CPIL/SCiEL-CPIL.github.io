"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# We need to define the integrand and the reference integration.
# The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
# where a_i = 1/(1+z*nodes[i]) and E(a) = H(a)/H0.
# The integral is actually: dL/(c/H0) = (1+z) * integral_0^z dz'/E(a(z'))
# Let x = z'/z. Then dz' = z dx.
# dL/(c/H0) = (1+z) * z * integral_0^1 dx / E(1/(1+z*x))
# The quadrature approximates the integral_0^1 dx / E(1/(1+z*x)) as sum_i weights[i] / E(a_i)
# where a_i = 1/(1+z*nodes[i]).
# So the quadrature rule approximates I(z) = integral_0^1 f(z, x) dx where f(z, x) = 1/E(1/(1+z*x)).
# The nodes are fixed in [0,1]. The weights are fixed.
# The error depends on z and the cosmology.

# We need to choose nodes and weights to minimize the worst-case relative error
# over a set of validation cosmologies and a range of z.

# Since we don't know the exact validation cosmologies, we need to use a representative set.
# The task mentions "spatially flat, time-varying dark-energy models" and "CPL equations".
# CPL parametrization: w(a) = w0 + w1*(1-a)
# E(a)^2 = (1-a)^-2 * a^(-3(1+w)) * exp(3*w1*(1-a))  (for flat universe with matter and DE)
# Actually, the standard form is:
# E(a)^2 = Omega_m * a^-3 + Omega_r * a^-4 + Omega_k * a^-2 + Omega_de * exp(-3*integral_1^a (1+w(a'))/a' da')
# For CPL with flat universe (Omega_k=0, Omega_m + Omega_de = 1):
# E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^-3 * exp(3*w1*(1-a))
# (assuming w0 is absorbed or we use the standard CPL form)
# More precisely:
# E(a)^2 = Omega_m * a^-3 + Omega_de * a^-3 * exp(-3*(1+w0)*integral_1^a da'/a' - 3*w1*integral_1^a da'/a' * (a'-1))
# integral_1^a da'/a' = ln(a)
# integral_1^a da'/a' * (a'-1) = integral_1^a (1 - 1/a') da' = [a' - ln(a')]_1^a = a - 1 - ln(a)
# So E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^-3 * exp(3*(1+w0)*ln(a) + 3*w1*(a-1-ln(a)))
#            = a^-3 * [Omega_m + (1-Omega_m) * a^(3*(1+w0)) * exp(3*w1*(a-1-ln(a)))]

# Let's define a function to compute E(a) for CPL parametrization.
def E_of_a(a, Omega_m, w0, w1):
    """Compute H(a)/H0 for CPL parametrization in a flat universe."""
    # a is array or scalar
    a = np.asarray(a)
    # Avoid log(0)
    a_safe = np.maximum(a, 1e-30)
    ln_a = np.log(a_safe)
    # Exponent term
    exponent = 3.0 * (1.0 + w0) * ln_a + 3.0 * w1 * (a_safe - 1.0 - ln_a)
    de_term = (1.0 - Omega_m) * np.exp(exponent)
    m_term = Omega_m
    return np.sqrt(a_safe ** (-3.0) * (m_term + de_term))

def integrand(x, z, Omega_m, w0, w1):
    """The integrand f(z, x) = 1/E(1/(1+z*x))"""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / E_of_a(a, Omega_m, w0, w1)

def quad_error(z, nodes, weights, Omega_m, w0, w1):
    """Compute the relative error for a given z and cosmology."""
    # Quadrature approximation
    a_quad = 1.0 / (1.0 + z * nodes)
    quad_val = np.sum(weights / E_of_a(a_quad, Omega_m, w0, w1))
    
    # Reference: adaptive integration
    from scipy.integrate import quad
    ref_val, _ = quad(lambda x: integrand(x, z, Omega_m, w0, w1), 0, 1, limit=200)
    
    if abs(ref_val) < 1e-30:
        return 0.0
    return abs(quad_val - ref_val) / abs(ref_val) * 100.0  # in percent

def worst_case_error(nodes, weights, cosmologies, z_values):
    """Compute the worst-case relative error over all cosmologies and z values."""
    max_err = 0.0
    for (Omega_m, w0, w1) in cosmologies:
        for z in z_values:
            err = quad_error(z, nodes, weights, Omega_m, w0, w1)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Define a set of representative cosmologies
    # Omega_m in [0.2, 0.4], w0 in [-1.5, 0.0], w1 in [-1.0, 1.0]
    cosmologies = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -1.0, 0.5),    # CPL
        (0.3, -0.8, 0.3),    # CPL
        (0.3, -1.2, 0.5),    # CPL
        (0.3, -0.9, -0.3),   # CPL
        (0.25, -1.0, 0.2),   # CPL
        (0.35, -1.1, 0.4),   # CPL
        (0.3, -0.7, 0.6),    # CPL
    ]
    
    # z values to test
    z_values = np.logspace(-2, 2, 20)  # 0.01 to 100
    
    # Initial guess: Gauss-Legendre quadrature nodes and weights
    # But Gauss-Legendre nodes are in [-1,1], we need [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    # Ensure nodes are strictly increasing and in (0,1)
    # leggauss nodes are already in (-1,1) and increasing
    # After mapping to (0,1), they are in (0,1) and increasing
    
    # We need to optimize nodes and weights subject to:
    # 1. nodes strictly increasing in (0,1)
    # 2. weights positive
    # 3. sum(weights) = 1
    
    # Use a parameterization that ensures these constraints.
    # nodes: use sigmoid of free parameters to map to (0,1)
    # weights: use softmax or exp to ensure positivity, then normalize
    
    def objective(params):
        # params: 8 node params + 8 weight params
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Map node params to (0,1) using sigmoid
        nodes = 1.0 / (1.0 + np.exp(-node_params))
        # Ensure strict ordering: we can use a cumulative sum of positive values
        # Or just use the sigmoid and hope the optimizer maintains ordering
        # Better: use a transformation that ensures ordering
        # Let's use: nodes[i] = sigmoid(node_params[i]) but add a penalty if not ordered
        
        # Check ordering
        if not np.all(np.diff(nodes) > 1e-12):
            return 1e6  # Penalty
        
        # Map weight params to positive values using exp
        weights_raw = np.exp(weight_params)
        weights = weights_raw / np.sum(weights_raw)
        
        # Compute worst-case error
        err = worst_case_error(nodes, weights, cosmologies, z_values)
        return err
    
    # Initial parameters
    node_params_init = np.log(nodes_init / (1.0 - nodes_init))  # inverse sigmoid
    weight_params_init = np.log(weights_init)  # inverse of exp
    
    # Optimize
    try:
        result = minimize(objective, np.concatenate([node_params_init, weight_params_init]),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() > deadline:
            # Use initial guess
            return {"nodes": nodes_init, "weights": weights_init}
        
        best_params = result.x
        node_params_best = best_params[:8]
        weight_params_best = best_params[8:16]
        
        nodes_best = 1.0 / (1.0 + np.exp(-node_params_best))
        weights_raw_best = np.exp(weight_params_best)
        weights_best = weights_raw_best / np.sum(weights_raw_best)
        
        # Ensure nodes are strictly increasing
        if not np.all(np.diff(nodes_best) > 1e-12):
            # Fallback to initial
            return {"nodes": nodes_init, "weights": weights_init}
        
        # Ensure weights sum to 1 within 1e-12
        weights_best = weights_best / np.sum(weights_best)
        
        return {"nodes": nodes_best, "weights": weights_best}
        
    except Exception:
        # Fallback to initial guess
        return {"nodes": nodes_init, "weights": weights_init}

# EVOLVE-BLOCK-END
