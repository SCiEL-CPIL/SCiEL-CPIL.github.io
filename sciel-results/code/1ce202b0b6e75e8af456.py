"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _cpl_e(a, wa0, wa1):
    """E(a) = H(a)/H0 for CPL model with Omega_m=0.3, Omega_L=0.7."""
    omega_m = 0.3
    omega_l = 0.7
    # E^2 = omega_m/a^3 + omega_l*(a/(1+a*wa1))^(3*(1+wa0))
    term_m = omega_m / (a**3)
    base = a / (1.0 + a * wa1)
    term_l = omega_l * (base ** (3.0 * (1.0 + wa0)))
    return np.sqrt(term_m + term_l)


def _integrate_midpoint(z_nodes, z_weights, wa0, wa1, n_dense=2000):
    """Compute the true integral using dense midpoint rule for reference."""
    # The integral is: ∫_0^z z'/(1+z') * 1/E(a(z')) dz'
    # where a(z') = 1/(1+z')
    # We change variable to a: z = 1/a - 1, dz = -da/a^2
    # Integral from z=0 to z=z_max becomes integral from a=1 to a=1/(1+z_max)
    # Let's just do it in z-space with dense sampling
    
    z_max = 2.0  # Cover up to z=2 for validation
    
    # Dense midpoint integration in z
    z_dense = np.linspace(0, z_max, n_dense + 1)
    z_mid = 0.5 * (z_dense[:-1] + z_dense[1:])
    dz = z_dense[1] - z_dense[0]
    
    a_mid = 1.0 / (1.0 + z_mid)
    E_mid = _cpl_e(a_mid, wa0, wa1)
    
    # Integrand: z/(1+z) * 1/E(a)
    integrand = z_mid / (1.0 + z_mid) / E_mid
    
    # True integral value
    true_val = np.sum(integrand) * dz
    
    # Quadrature approximation
    # Our quadrature: sum_i w_i * f(z_i) where f(z) = z/(1+z)/E(a(z))
    # But the formula says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # Wait, let me re-read: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # Hmm, that's a bit unusual. Let me think about this more carefully.
    
    # Actually, the luminosity distance is:
    # d_L = (c/H0) * (1+z) * ∫_0^z dz' / E(a(z'))
    # So d_L/(c/H0) = (1+z) * ∫_0^z dz' / E(a(z'))
    
    # The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This means the quadrature is approximating ∫_0^z dz'/E(a(z')) as z*sum_i weights[i]/E(a_i)
    
    # Wait, that doesn't seem right dimensionally. Let me re-read.
    
    # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    
    # So the quadrature rule is for the integral ∫_0^z dz'/E(a(z')), and it's being
    # approximated as z * sum_i w_i / E(a(z*n_i))
    
    # This is a quadrature of the form: ∫_0^z g(z') dz' ≈ z * sum_i w_i * g(z * n_i)
    # where g(z') = 1/E(a(z'))
    
    # This is equivalent to: let u = z'/z, then z' = z*u, dz' = z*du
    # ∫_0^z g(z') dz' = z * ∫_0^1 g(z*u) du
    # So the quadrature approximates ∫_0^1 h(u) du where h(u) = g(z*u) = 1/E(a(z*u))
    # and a(z*u) = 1/(1+z*u)
    
    # So the nodes are in [0,1] and represent u values, and we're approximating
    # ∫_0^1 1/E(a(z*u)) du using sum_i w_i / E(a(z*n_i))
    
    # The relative error would be:
    # |approx - true| / true
    
    # For a given z, the true integral is ∫_0^1 1/E(a(z*u)) du
    # The approximation is sum_i w_i / E(a(z*n_i))
    
    # Let me compute this properly
    
    # For a test z value, compute true and approx
    z_test = 1.0  # We'll optimize over a range of z and cosmologies
    
    return true_val  # placeholder, need to restructure


def _objective(params, z_samples, wa0_samples, wa1_samples):
    """Compute sum of squared log-errors over a grid of z and cosmologies."""
    nodes = params[:8]
    log_weights = params[8:]
    weights = np.exp(log_weights)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    total_error = 0.0
    count = 0
    
    n_dense = 500
    
    for z in z_samples:
        for wa0 in wa0_samples:
            for wa1 in wa1_samples:
                # True integral: ∫_0^1 1/E(a(z*u)) du
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
                    rel_err = (approx_val - true_val) / true_val
                    total_error += np.log1p(rel_err**2)
                    count += 1
    
    return total_error / max(count, 1)


def construct():
    # Define a grid of z values and CPL parameters to optimize over
    z_samples = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    wa0_samples = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    wa1_samples = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    
    # Initial guess: midpoint rule
    nodes_init = (np.arange(8) + 0.5) / 8.0
    weights_init = np.full(8, 1.0 / 8.0)
    log_weights_init = np.log(weights_init)
    
    params_init = np.concatenate([nodes_init, log_weights_init])
    
    # Bounds: nodes in (0, 1) strictly increasing, weights positive
    # We'll use a transformation to enforce ordering
    # Actually, let's just optimize with constraints handled by the optimizer
    
    # Use L-BFGS-B
    res = minimize(
        _objective,
        params_init,
        args=(z_samples, wa0_samples, wa1_samples),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14}
    )
    
    # Extract optimized parameters
    nodes_opt = res.x[:8]
    log_weights_opt = res.x[8:]
    weights_opt = np.exp(log_weights_opt)
    weights_opt = weights_opt / np.sum(weights_opt)
    
    # Ensure nodes are strictly increasing and in [0, 1]
    # Sort if needed
    order = np.argsort(nodes_opt)
    nodes_opt = nodes_opt[order]
    weights_opt = weights_opt[order]
    
    # Clip nodes to [0, 1]
    nodes_opt = np.clip(nodes_opt, 1e-6, 1.0 - 1e-6)
    
    # Ensure strict monotonicity
    for i in range(1, 8):
        if nodes_opt[i] <= nodes_opt[i-1]:
            nodes_opt[i] = nodes_opt[i-1] + 1e-6
    
    # Final normalization of weights
    weights_opt = weights_opt / np.sum(weights_opt)
    
    return {"nodes": nodes_opt, "weights": weights_opt}
# EVOLVE-BLOCK-END
