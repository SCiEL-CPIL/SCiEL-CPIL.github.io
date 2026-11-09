"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    # E(a) = (a^(-3(1+w0)) * (1 + w0 + wa * (1 - a))^(wa/(1+w0)))
    # But more standard: E(z) = (1+z)^3 * (1 + w0 + wa*z)^(wa/(1+w0))
    # In terms of a: z = 1/a - 1
    # E(a) = a^(-3) * (1 + w0 + wa*(1/a - 1))^(wa/(1+w0))
    # Simplify the term inside: 1 + w0 + wa*(1/a - 1) = 1 + w0 + wa/a - wa = (1 + w0 - wa) + wa/a
    # So E(a) = a^(-3) * ((1 + w0 - wa) + wa/a)^(wa/(1+w0))
    #          = a^(-3) * ((a*(1+w0-wa) + wa)/a)^(wa/(1+w0))
    #          = a^(-3) * (a*(1+w0-wa) + wa)^(wa/(1+w0)) * a^(-wa/(1+w0))
    #          = a^(-3 - wa/(1+w0)) * (a*(1+w0-wa) + wa)^(wa/(1+w0))
    
    # Alternative simpler form:
    # E(a) = (a^(-3(1+w0)) * (1 + w0 + wa*(1-a))^(wa/(1+w0)))
    # Let's use the z-based formula converted to a:
    # E(a) = a^(-3) * (1 + w0 + wa*(1/a - 1))^(wa/(1+w0))
    
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    # Handle potential negative base if parameters are extreme, but for reasonable CPL it should be positive
    if np.any(base <= 0):
        # Fallback: use a more robust form or clip
        base = np.maximum(base, 1e-10)
    exponent = wa / (1.0 + w0)
    return a**(-3.0) * base**exponent


def _luminosity_distance_integrand(z, w0, wa):
    """
    Integrand for dL/(c/H0) = (1+z) * integral_0^z dz' / E(z')
    Actually, dL = c/H0 * (1+z) * integral_0^z dz' / E(z')
    So the quadrature approximates: (1+z) * sum_i w_i / E(z_i)
    where the sum approximates integral_0^z dz'/E(z')
    
    Wait, the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    This implies the quadrature is for the integral I(z) = integral_0^z dz'/E(z')
    approximated by z * sum_i w_i / E(z_i)
    So the quadrature rule is for the function f(z') = 1/E(z') over [0, z]
    using the form: integral_0^z f(z') dz' ≈ z * sum_i w_i f(z_i)
    with nodes z_i = z * nodes[i] where nodes[i] are in [0,1]
    
    Let's re-read: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    So a_i = 1/(1 + z*nodes[i]) means z_i = z*nodes[i]
    And E(a_i) = E(z_i) where E is H/H0
    So the quadrature approximates integral_0^z dz'/E(z') by z * sum_i w_i / E(z_i)
    This is a quadrature rule of the form: integral_0^z f(t) dt ≈ z * sum_i w_i f(z*t_i)
    where t_i = nodes[i] are fixed in [0,1]
    """
    pass


def _exact_integral(z, w0, wa):
    """Compute exact integral_0^z dz'/E(z') using adaptive quadrature."""
    from scipy.integrate import quad
    
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return 1.0 / _E_cpl(a, w0, wa)
    
    result, error = quad(integrand, 0, z, limit=200)
    return result


def _quadrature_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    return z * np.sum(weights / E_vals)


def _relative_error(z, nodes, weights, w0, wa):
    """Compute relative error of quadrature for given z and cosmology."""
    exact = _exact_integral(z, w0, wa)
    approx = _quadrature_value(z, nodes, weights, w0, wa)
    if abs(exact) < 1e-15:
        return 0.0
    return abs((approx - exact) / exact)


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    # Representative CPL models (w0, wa)
    # These should cover a reasonable range of dark energy models
    models = [
        (0.0, 0.0),    # Lambda-CDM
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, -0.5),  # Quintessence-like
        (-1.0, 0.5),   # Phantom-like
        (-0.8, -0.5),  # Another quintessence
        (-1.2, 0.3),   # More phantom
        (-0.3, -0.3),  # Mild quintessence
        (-1.0, -0.5),  # Mixed
    ]
    
    # Test redshifts
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    
    # Initial guess: Gauss-Legendre nodes and weights
    # But Gauss-Legendre is for integral over [0,1] with weight 1
    # Our quadrature is: integral_0^z f(t) dt ≈ z * sum w_i f(z*t_i)
    # This is equivalent to: integral_0^1 f(z*u) du ≈ sum w_i f(z*u_i)
    # So it's a Gauss-Legendre rule on [0,1] for the function g(u) = f(z*u)
    # Since the nodes and weights must be universal, we need a rule that works
    # well for all z and all cosmologies
    
    # Start with Gauss-Legendre nodes on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)  # Map from [-1,1] to [0,1]
    weights0 = 0.5 * gl_weights
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes0 = np.clip(nodes0, 1e-10, 1.0 - 1e-10)
    nodes0 = np.sort(nodes0)
    
    # Optimize to minimize worst-case relative error
    def objective(params):
        if time.time() > deadline:
            return 1e10
        # params: 16 values = 8 nodes + 8 weights
        # Use log-weights to ensure positivity
        nodes = params[:8]
        log_weights = params[8:]
        
        # Enforce nodes in (0,1) and strictly increasing
        # Use a soft constraint via penalty
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        
        weights = np.exp(log_weights)
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute worst-case relative error
        max_err = 0.0
        for w0, wa in models:
            for z in z_test:
                err = _relative_error(z, nodes, weights, w0, wa)
                if err > max_err:
                    max_err = err
                if time.time() > deadline:
                    return 1e10
        
        return max_err
    
    # Initial parameters
    params0 = np.concatenate([nodes0, np.log(weights0)])
    
    # Use Nelder-Mead for optimization (doesn't require gradient)
    try:
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.success or result.fun < 1e10:
            nodes_opt = result.x[:8]
            log_weights_opt = result.x[8:]
            weights_opt = np.exp(log_weights_opt)
            weights_opt = weights_opt / np.sum(weights_opt)
            
            # Ensure nodes are strictly increasing and in (0,1)
            if np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and np.all(np.diff(nodes_opt) > 0):
                return {"nodes": nodes_opt, "weights": weights_opt}
    except Exception:
        pass
    
    # Fallback: return Gauss-Legendre
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END
