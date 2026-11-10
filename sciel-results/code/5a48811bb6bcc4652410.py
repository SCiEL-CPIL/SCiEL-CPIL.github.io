"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Uses Gauss-Legendre quadrature as the starting point, then refines via optimization
    to minimize worst-case relative error across representative CPL dark energy models.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s wall-clock limit
    
    def E_of_a(a, w0, w1):
        """E(a) = H(a)/H0 for flat universe with CPL dark energy."""
        # a = 1/(1+z), so z = 1/a - 1
        # E^2(a) = Omega_m * a^-3 + Omega_r * a^-4 + Omega_de * a^(-3*(1+w0+w1*(1/a-1)))
        Omega_m = 0.3
        Omega_r = 1e-5  # small radiation component
        Omega_de = 1.0 - Omega_m - Omega_r
        
        # CPL: w(a) = w0 + w1*(1-a)/a  ... wait, standard CPL is w(z) = w0 + w1*z
        # In terms of a: z = 1/a - 1, so w(a) = w0 + w1*(1/a - 1)
        # The DE density scales as a^(-3*(1+w)) for constant w
        # For CPL, we use the approximation: rho_de(a) ~ a^(-3*(1+w0+w1*(1-a)/a))
        # Actually, the standard form is:
        # E^2(a) = Omega_m a^-3 + Omega_de * a^(-3*(1+w0+w1*(1-a)/a))
        # Let me use the more standard CPL parameterization
        
        z = 1.0 / a - 1.0
        w = w0 + w1 * z
        # For CPL, the DE scale factor dependence is:
        # rho_de(a) = rho_de0 * a^(-3*(1+w0+w1*(1-a)/a))  (approximate)
        # More precisely, using the CPL formula:
        # rho_de(a)/rho_de0 = a^(-3*(1+w0+w1*(1-a)/a))
        
        exponent = -3.0 * (1.0 + w0 + w1 * (1.0 - a) / a)
        de_term = Omega_de * (a ** exponent)
        
        m_term = Omega_m * a**(-3)
        r_term = Omega_r * a**(-4)
        
        return np.sqrt(m_term + r_term + de_term)
    
    def reference_integral(z_max, w0, w1, n_steps=100000):
        """Adaptive-like reference integral using fine trapezoidal rule."""
        # dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
        # Change variable to a = 1/(1+z'), so z' = 1/a - 1, dz' = -da/a^2
        # When z'=0, a=1; when z'=z_max, a=1/(1+z_max)
        # Integral becomes: integral_{a_min}^{1} (1/a) * (1/E(a)) * (da/a^2)
        # Wait, let me redo this carefully.
        
        # dL/(c/H0) = (1+z) * integral_0^z dz' / E(z')
        # Let a = 1/(1+z'), so z' = 1/a - 1, dz' = -da/a^2
        # When z' = 0, a = 1
        # When z' = z, a = 1/(1+z)
        
        # integral_0^z dz'/E(z') = integral_1^{1/(1+z)} (-da/a^2) / E(a)
        # = integral_{1/(1+z)}^1 da / (a^2 * E(a))
        
        # So dL/(c/H0) = (1+z) * integral_{1/(1+z)}^1 da / (a^2 * E(a))
        
        a_min = 1.0 / (1.0 + z_max)
        a_max = 1.0
        a_vals = np.linspace(a_min, a_max, n_steps)
        E_vals = E_of_a(a_vals, w0, w1)
        integrand = 1.0 / (a_vals**2 * E_vals)
        
        # Trapezoidal rule
        integral = np.trapz(integrand, a_vals)
        return (1.0 + z_max) * integral
    
    def quadrature_value(nodes, weights, z_max, w0, w1):
        """Compute quadrature approximation of dL/(c/H0)."""
        a_i = 1.0 / (1.0 + z_max * nodes)
        E_i = E_of_a(a_i, w0, w1)
        # The integrand in the quadrature formula:
        # dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
        # Wait, I need to check the formula again.
        
        # From the task: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
        
        # So the quadrature is: (1+z) * z * sum_i w_i / E(1/(1+z*x_i))
        
        val = (1.0 + z_max) * z_max * np.sum(weights / E_i)
        return val
    
    def worst_relative_error(nodes, weights, z_list, w0_list, w1_list):
        """Compute worst relative error across test cases."""
        max_err = 0.0
        for z_max, w0, w1 in zip(z_list, w0_list, w1_list):
            ref = reference_integral(z_max, w0, w1)
            quad = quadrature_value(nodes, weights, z_max, w0, w1)
            if ref > 1e-10:
                rel_err = abs(quad - ref) / ref
                max_err = max(max_err, rel_err)
        return max_err
    
    # Define test cosmologies and redshifts for optimization
    # These are representative CPL models
    w0_list = [-1.0, -1.0, -1.0, -0.9, -1.1, -0.8, -1.2, -1.0]
    w1_list = [0.0, 0.5, -0.5, 0.3, 0.4, 0.2, -0.3, 1.0]
    z_list = [0.5, 1.0, 2.0, 3.0, 1.0, 0.5, 2.0, 1.5]
    
    # Start with Gauss-Legendre 8-point rule
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map from [-1, 1] to [0, 1]
    nodes_gl = (x_gl + 1.0) / 2.0
    nodes_gl = np.sort(nodes_gl)
    w_gl_mapped = w_gl / 2.0
    # Reorder weights to match sorted nodes
    idx = np.argsort(x_gl)
    nodes_gl = (x_gl[idx] + 1.0) / 2.0
    w_gl_mapped = w_gl[idx] / 2.0
    # Normalize weights to sum to 1
    w_gl_mapped = w_gl_mapped / np.sum(w_gl_mapped)
    
    best_nodes = nodes_gl.copy()
    best_weights = w_gl_mapped.copy()
    best_err = worst_relative_error(best_nodes, best_weights, z_list, w0_list, w1_list)
    
    # Optimization: minimize worst-case error
    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        n = len(params) // 2
        nodes = params[:n]
        weights = params[n:]
        
        # Enforce constraints: nodes in (0,1) strictly increasing, weights positive
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        return worst_relative_error(nodes, weights, z_list, w0_list, w1_list)
    
    # Use Nelder-Mead for derivative-free optimization
    x0 = np.concatenate([nodes_gl, w_gl_mapped])
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    if time.time() < DEADLINE - 5.0:
        # Try a second optimization from a different starting point
        # Start from equally spaced nodes
        n = 8
        nodes_eq = np.linspace(0.1, 0.9, n)
        weights_eq = np.full(n, 1.0 / n)
        x0_eq = np.concatenate([nodes_eq, weights_eq])
        
        result2 = minimize(objective, x0_eq, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        
        if result2.fun < best_err:
            best_err = result2.fun
            best_nodes = result2.x[:n].copy()
            best_weights = result2.x[n:].copy()
    
    # Extract best solution
    if result.fun < best_err:
        best_err = result.fun
        best_nodes = result.x[:8].copy()
        best_weights = result.x[8:].copy()
    
    # Ensure constraints are satisfied
    best_nodes = np.sort(best_nodes)
    # Clip nodes to be strictly inside (0,1)
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    
    # Ensure positive weights and normalize
    best_weights = np.abs(best_weights)
    if np.any(best_weights <= 0):
        best_weights = np.full(8, 1.0/8)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
