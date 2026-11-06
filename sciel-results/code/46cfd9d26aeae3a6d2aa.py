"""Optimized quadrature rule for luminosity distance using CPL cosmologies."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Construct an 8-node positive quadrature rule for luminosity distance."""
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin for 90s limit
    
    # The integrand for luminosity distance in dimensionless form
    # dL/(c/H0) = (1+z) * integral_0^z dz' / E(a(z'))
    # where a = 1/(1+z'), E(a) = H(a)/H0
    #
    # In terms of a (scale factor), with a = 1/(1+z):
    # dL/(c/H0) = integral_0^1 da / (a^2 * E(a))
    # 
    # But the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    #
    # Let me re-read: the formula given is:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # with a_i = 1/(1+z*nodes[i])
    #
    # So the quadrature approximates: (1+z)*z * sum w_i / E(a_i)
    # where the nodes are in [0,1] and represent z' values scaled by z.
    #
    # Actually, let's think about this more carefully. The standard luminosity distance is:
    # dL = (1+z) * c/H0 * integral_0^z dz' / E(z')
    #
    # If we substitute z' = z * t where t in [0,1], dz' = z * dt:
    # dL/(c/H0) = (1+z) * integral_0^1 z * dt / E(z*t)
    #           = (1+z)*z * integral_0^1 dt / E(z*t)
    #
    # So the integral to approximate is: integral_0^1 dt / E(z*t)
    # and the quadrature is: sum_i w_i / E(z * nodes[i])
    # where nodes[i] are in [0,1] and E(z*t) = H(a)/H0 with a = 1/(1+z*t)
    
    # For CPL parametrization:
    # w(a) = w0 + wa*(a-1)
    # E(a)^2 = Omega_m*a^-3 + Omega_r*a^-4 + Omega_k*a^-2 + Omega_DE * exp(3*int_1^a (1+w(a'))/a' da')
    #
    # For flat universe (Omega_k=0):
    # Omega_DE = 1 - Omega_m - Omega_r
    #
    # The integral in the exponent: int_1^a (1+w(a'))/a' da' where w(a') = w0 + wa*(a'-1)
    # = int_1^a [1 + w0 + wa*(a'-1)]/a' da'
    # = (1+w0) * ln(a) + wa * int_1^a (a'-1)/a' da'
    # = (1+w0)*ln(a) + wa * int_1^a (1 - 1/a') da'
    # = (1+w0)*ln(a) + wa * [a' - ln(a')]_1^a
    # = (1+w0)*ln(a) + wa * [(a - ln(a)) - (1 - 0)]
    # = (1+w0)*ln(a) + wa * (a - 1 - ln(a))
    
    def E_squared(a, Omega_m, Omega_r, w0, wa):
        """Compute E(a)^2 for CPL model."""
        Omega_DE = 1.0 - Omega_m - Omega_r
        # Handle a near 0 carefully
        a_safe = np.maximum(a, 1e-30)
        ln_a = np.log(a_safe)
        exponent = (1.0 + w0) * ln_a + wa * (a_safe - 1.0 - ln_a)
        DE_term = Omega_DE * np.exp(3.0 * exponent)
        return Omega_m * a_safe**-3 + Omega_r * a_safe**-4 + DE_term
    
    def integrand(t, z, Omega_m, Omega_r, w0, wa):
        """The integrand 1/E(z*t) where t in [0,1]."""
        a = 1.0 / (1.0 + z * t)
        E2 = E_squared(a, Omega_m, Omega_r, w0, wa)
        return 1.0 / np.sqrt(np.maximum(E2, 1e-30))
    
    def quadrature_error(nodes, weights, z, Omega_m, Omega_r, w0, wa):
        """Compute relative error of quadrature vs true integral."""
        # Quadrature approximation
        a_vals = 1.0 / (1.0 + z * nodes)
        E2_vals = E_squared(a_vals, Omega_m, Omega_r, w0, wa)
        quad_val = np.sum(weights / np.sqrt(np.maximum(E2_vals, 1e-30)))
        
        # True integral using adaptive quadrature (simple Simpson with many points)
        n_pts = 10000
        t_grid = np.linspace(0, 1, n_pts)
        f_vals = integrand(t_grid, z, Omega_m, Omega_r, w0, wa)
        true_val = np.trapz(f_vals, t_grid)
        
        if true_val < 1e-15:
            return 0.0
        return abs(quad_val - true_val) / true_val
    
    # Define a set of CPL cosmologies to optimize against
    # These should span a reasonable range of parameters
    cosmologies = [
        # (Omega_m, Omega_r, w0, wa)
        (0.3, 0.0, -1.0, 0.0),       # LCDM
        (0.3, 0.0, -1.0, 0.5),       # CPL with positive wa
        (0.3, 0.0, -1.0, -0.5),      # CPL with negative wa
        (0.27, 0.0, -0.9, 0.3),      # Different Omega_m
        (0.32, 0.0, -1.1, 0.2),      # Different w0
        (0.3, 0.0, -0.8, 0.0),       # Different w0
        (0.25, 0.0, -1.0, 0.4),      # Lower Omega_m
        (0.35, 0.0, -1.2, -0.3),     # Higher Omega_m, negative wa
        (0.3, 0.0, -1.0, 1.0),       # Strong CPL
        (0.3, 0.0, -1.0, -1.0),      # Strong negative CPL
        (0.3, 0.0, -0.5, 0.0),       # Much different w0
        (0.3, 0.0, -1.5, 0.5),       # Phantom-ish
    ]
    
    # Test redshifts
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    def objective(params):
        """Worst-case relative error across all test cosmologies and redshifts."""
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce ordering: nodes strictly increasing in [0,1]
        # Use soft constraints via penalty
        node_diffs = np.diff(nodes)
        if np.any(node_diffs <= 0):
            return 1e10
        if nodes[0] < 0 or nodes[-1] > 1:
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        max_err = 0.0
        for (Om, Or, w0, wa) in cosmologies:
            for z in z_test:
                err = quadrature_error(nodes, weights, z, Om, Or, w0, wa)
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Initial guess: midpoint rule
    n = 8
    nodes0 = (np.arange(n) + 0.5) / n
    weights0 = np.full(n, 1.0 / n)
    x0 = np.concatenate([nodes0, weights0])
    
    # Use SLSQP with constraints
    # Constraints:
    # 1. nodes in [0, 1] (with small margins)
    # 2. nodes strictly increasing
    # 3. weights positive
    # 4. sum(weights) = 1
    
    constraints = []
    # Sum of weights = 1
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0})
    
    # Bounds: nodes in [0.01, 0.99], weights in [0.001, 1]
    bounds = []
    for i in range(8):
        bounds.append((0.01, 0.99))  # nodes
    for i in range(8):
        bounds.append((0.001, 1.0))  # weights
    
    # Try optimization with time check
    best_x = x0.copy()
    best_err = objective(x0)
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        
        if result.success or result.fun < best_err:
            # Verify constraints
            res_nodes = result.x[:8]
            res_weights = result.x[8:]
            if np.all(np.diff(res_nodes) > 0) and np.all(res_weights > 0):
                best_x = result.x
                best_err = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 60.0:
        # Time out, return best so far
        pass
    
    # If we have time, try a second optimization with different initial conditions
    if time.time() < deadline - 10.0:
        # Try Gauss-Legendre-like nodes
        try:
            # Use Legendre polynomial roots as initial nodes (mapped to [0,1])
            from numpy.polynomial.legendre import leggauss
            x_leg, w_leg = leggauss(8)
            nodes_leg = (x_leg + 1) / 2  # map from [-1,1] to [0,1]
            weights_leg = w_leg / 2.0    # scale weights
            x0_leg = np.concatenate([nodes_leg, weights_leg])
            
            result2 = minimize(objective, x0_leg, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
            
            if result2.success or result2.fun < best_err:
                res_nodes2 = result2.x[:8]
                res_weights2 = result2.x[8:]
                if np.all(np.diff(res_nodes2) > 0) and np.all(res_weights2 > 0):
                    best_x = result2.x
                    best_err = result2.fun
        except Exception:
            pass
    
    # Extract final result
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Ensure weights sum to 1 exactly (within tolerance)
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > 1e-15:
        weights = weights / weight_sum
    
    # Final validation
    if not np.all(np.diff(nodes) > 0):
        # Fallback to midpoint
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    if not np.all(weights > 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
