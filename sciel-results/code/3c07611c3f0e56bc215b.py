"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    The integral is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i]) and E(a) = H(a)/H0.
    
    We minimize the worst-case relative error across a family of CPL cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # CPL model: E(a) = a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(3*wa)
    # For flat universe with dark energy: E^2 = (1-a)^-6 * (1 + wa*(1-a))^(3*wa) + matter term
    # Actually for flat: E^2(a) = a^(-6) * Omega_m + a^(-3(1+w0)) * (1+wa*(1-a))^(3*wa) * Omega_DE
    # Let's use the standard CPL form:
    # E(z) = (1+z)^(3(1+w0+wa)) / (1+wa*z)^(3*wa)
    # In terms of a: E(a) = a^(-3(1+w0+wa)) / (1+wa*(1-a))^(3*wa)
    
    def E_cpl(a, w0, wa):
        """CPL dark energy E(a) = H(a)/H0 for flat universe."""
        # For simplicity, use the standard CPL form with Omega_m = 0.3, Omega_DE = 0.7
        # E^2(a) = Omega_m * a^(-6) + Omega_DE * a^(-3(1+w0)) * (1+wa*(1-a))^(3*wa)
        # But the task says the evaluator computes integrands itself, so we just need
        # nodes and weights that work well. Let's use a representative family.
        Omega_m = 0.3
        Omega_DE = 0.7
        term_m = Omega_m * np.power(a, -6)
        term_de = Omega_DE * np.power(a, -3*(1+w0)) * np.power(1+wa*(1-a), 3*wa)
        return np.sqrt(term_m + term_de)
    
    def integrand(z, w0, wa):
        """Compute (1+z)*z/E(a) for given redshift z and CPL parameters."""
        a = 1.0 / (1.0 + z)
        return (1.0 + z) * z / E_cpl(a, w0, wa)
    
    def quadrature(z, nodes, weights, w0, wa):
        """Compute the quadrature approximation of the luminosity distance integral."""
        a_nodes = 1.0 / (1.0 + z * nodes)
        E_vals = E_cpl(a_nodes, w0, wa)
        return np.sum(weights / E_vals)
    
    def true_integral(z, w0, wa, n_points=1000):
        """Compute the true integral using adaptive quadrature."""
        from scipy.integrate import quad
        f = lambda xi: (1.0 + z*xi) * z * xi / E_cpl(1.0/(1.0+z*xi), w0, wa)
        result, _ = quad(f, 0, 1, limit=n_points)
        return result
    
    def worst_case_error(nodes, weights, w0_range, wa_range, z_range):
        """Compute worst-case relative error over a grid of cosmologies and redshifts."""
        max_err = 0.0
        # Sample a few representative cosmologies and redshifts
        for w0 in w0_range:
            for wa in wa_range:
                for z in z_range:
                    if z < 1e-6:
                        continue
                    approx = quadrature(z, nodes, weights, w0, wa)
                    exact = true_integral(z, w0, wa)
                    if abs(exact) > 1e-12:
                        rel_err = abs(approx - exact) / abs(exact)
                        max_err = max(max_err, rel_err)
        return max_err
    
    # Initial guess: Gauss-Legendre-like nodes
    nodes0 = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.90])
    weights0 = np.array([0.05, 0.10, 0.12, 0.13, 0.13, 0.13, 0.15, 0.19])
    
    # Normalize weights to sum to 1
    weights0 = weights0 / np.sum(weights0)
    
    # Define the objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10  # Timeout penalty
        
        x = params
        nodes = x[:8]
        weights = x[8:]
        
        # Ensure nodes are strictly increasing in [0,1]
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e10
        if np.any(np.diff(nodes) <= 1e-8):
            return 1e10
        
        # Ensure weights are positive and sum to 1
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10
        
        # Compute worst-case error
        w0_range = np.array([-1.0, -0.5, -0.3, 0.0, 0.5])
        wa_range = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        z_range = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
        
        return worst_case_error(nodes, weights, w0_range, wa_range, z_range)
    
    # Use differential evolution for global optimization
    bounds = [
        (1e-4, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
        (0.4, 0.5), (0.5, 0.6), (0.6, 0.85), (0.85, 0.999)
    ] + [(1e-4, 0.5)] * 8
    
    x0 = np.concatenate([nodes0, weights0])
    
    result = differential_evolution(
        objective, bounds, x0=x0, maxiter=50, popsize=15,
        tol=1e-6, seed=42, polish=True
    )
    
    if result.success or result.fun < 1e9:
        best_params = result.x
        nodes = best_params[:8]
        weights = best_params[8:]
        
        # Final normalization
        weights = weights / np.sum(weights)
        
        # Ensure strict increasing
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1] + 1e-8:
                nodes[i] = nodes[i-1] + 1e-6
        
        # Clip to valid range
        nodes[0] = max(nodes[0], 1e-6)
        nodes[-1] = min(nodes[-1], 1 - 1e-6)
        
        return {"nodes": nodes, "weights": weights}
    else:
        # Fallback: return the initial guess
        return {"nodes": nodes0, "weights": weights0}

# EVOLVE-BLOCK-END
