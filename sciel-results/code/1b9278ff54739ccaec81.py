"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.optimize import linprog


def construct():
    """
    Construct an 8-node positive quadrature rule for the luminosity distance integral.
    
    Uses a two-stage optimization:
    1. Optimize nodes using SLSQP with a diverse cosmology sample
    2. For optimized nodes, find optimal weights via constrained least squares
    
    The integral is: dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i)
    where a_i = 1/(1+z*x_i), x_i are nodes in [0,1], and E(a) = H(a)/H0.
    """
    
    n = 8
    
    # Define a diverse set of representative cosmologies for fitting
    # w0, wa pairs covering a broad range of physical models
    cosmologies = [
        (0.0, 0.0),     # LCDM-like
        (-1.0, 0.0),    # Cosmological constant
        (-0.5, 0.5),    # Phantom-ish
        (-1.5, 1.0),    # Strongly evolving
        (0.5, -0.5),    # Quintessence-like
        (-1.0, 0.5),    # Mildly evolving
        (-0.8, -0.2),   # Near LCDM
        (1.0, 0.0),     # Exotic
        (-2.0, 2.0),    # Very strong evolution
        (0.0, 1.0),     # Matter-dominated with evolution
        (-0.3, 0.3),    # Moderate
        (-1.2, -0.3),   # Another variant
        (-1.0, -0.5),   # Negative evolution
        (-0.6, 0.0),    # Flat dark energy
        (0.2, 0.2),     # Positive evolution
        (-1.5, 0.0),    # Strong negative w0
        (-0.2, 0.8),    # Moderate positive wa
        (0.8, -0.3),    # Exotic positive w0
    ]
    
    # Test redshifts - more comprehensive coverage
    z_values = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0, 200.0, 300.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        Omega_m = 0.3
        Omega_r = 0.0001
        Omega_de = 1.0 - Omega_m - Omega_r
        
        de_part = Omega_de * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1-a))
        matter_part = Omega_m * a**(-3)
        radiation_part = Omega_r * a**(-4)
        
        return np.sqrt(matter_part + radiation_part + de_part)
    
    def integrand_z(z, w0, wa):
        """The integrand for the luminosity distance integral."""
        a = 1.0/(1.0+z)
        return z / E_of_a(a, w0, wa)
    
    def exact_integral(z, w0, wa):
        """Compute the exact integral using adaptive quadrature."""
        if z < 1e-10:
            return 0.0
        result, _ = quad(integrand_z, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
        return result
    
    def quadrature_integral(z, w0, wa, nodes, weights):
        """Compute the quadrature approximation."""
        a_vals = 1.0/(1.0 + z*nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return z * np.sum(weights * nodes / E_vals)
    
    def relative_error(z, w0, wa, nodes, weights):
        """Compute relative error for a given cosmology and redshift."""
        exact = exact_integral(z, w0, wa)
        approx = quadrature_integral(z, w0, wa, nodes, weights)
        if abs(exact) < 1e-15:
            return 0.0
        return abs(approx - exact) / abs(exact)
    
    def worst_case_error(nodes, weights):
        """Compute the worst-case relative error over all test cases."""
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_values:
                err = relative_error(z, w0, wa, nodes, weights)
                if err > max_err:
                    max_err = err
        return max_err
    
    def objective_nodes(node_params):
        """
        Objective for node optimization.
        Given nodes, find optimal weights via constrained least squares,
        then return the worst-case error.
        """
        nodes = node_params
        
        # Enforce node constraints
        if np.any(nodes <= 1e-8) or np.any(nodes >= 1.0 - 1e-8):
            return 1e6
        if np.any(np.diff(nodes) <= 1e-6):
            return 1e6
        
        # Find optimal weights for these nodes
        # We want to minimize max relative error, which is non-convex
        # Instead, we minimize sum of squared relative errors as a proxy
        
        # Build the system: for each (z, w0, wa), we have
        # approx = z * sum_i w_i * nodes_i / E(a_i)
        # exact = I(z, w0, wa)
        # We want approx ≈ exact
        
        # Collect all test cases
        all_z = []
        all_w0 = []
        all_wa = []
        all_exact = []
        
        for w0, wa in cosmologies:
            for z in z_values:
                exact = exact_integral(z, w0, wa)
                if abs(exact) > 1e-12:
                    all_z.append(z)
                    all_w0.append(w0)
                    all_wa.append(wa)
                    all_exact.append(exact)
        
        m = len(all_z)
        
        # Build matrix A where A[j,i] = z_j * nodes_i / E(a_i(z_j))
        # and b[j] = exact_j
        # We want A @ w ≈ b, with w > 0 and sum(w) = 1
        
        A = np.zeros((m, n))
        b = np.zeros(m)
        
        for j in range(m):
            z_j = all_z[j]
            w0_j = all_w0[j]
            wa_j = all_wa[j]
            exact_j = all_exact[j]
            
            a_vals = 1.0/(1.0 + z_j * nodes)
            E_vals = E_of_a(a_vals, w0_j, wa_j)
            A[j, :] = z_j * nodes / E_vals
            b[j] = exact_j
        
        # Normalize to make the problem better conditioned
        # Divide each row by b[j] to get relative error
        mask = np.abs(b) > 1e-12
        A_rel = A[mask, :] / b[mask, :][:, None]
        b_rel = np.ones(np.sum(mask))
        
        # Solve: minimize ||A_rel @ w - b_rel||^2 subject to w > 0, sum(w) = 1
        # Using linear programming with a slack variable for the equality constraint
        
        # Actually, let's use scipy.optimize.minimize with SLSQP for this
        from scipy.optimize import minimize as min_opt
        
        def weight_objective(w):
            if np.any(w <= 0):
                return 1e10
            if abs(np.sum(w) - 1.0) > 1e-8:
                return 1e10
            residuals = A_rel @ w - b_rel
            return np.sum(residuals**2)
        
        # Initial guess: uniform weights
        w_init = np.ones(n) / n
        
        try:
            w_result = min_opt(weight_objective, w_init, method='SLSQP',
                              bounds=[(1e-10, 1.0)] * n,
                              constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
                              options={'maxiter': 500, 'ftol': 1e-12})
            
            if w_result.success and np.all(w_result.x > 0):
                weights = w_result.x
                weights = weights / np.sum(weights)
            else:
                # Fallback to uniform
                weights = np.ones(n) / n
        except:
            weights = np.ones(n) / n
        
        # Compute worst-case error with these weights
        max_err = 0.0
        for j in range(m):
            z_j = all_z[j]
            w0_j = all_w0[j]
            wa_j = all_wa[j]
            err = relative_error(z_j, w0_j, wa_j, nodes, weights)
            if err > max_err:
                max_err = err
        
        return max_err * 100.0
    
    # Initial guesses for nodes
    x_gl = np.array([
        -0.9602898564975362,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425002680,
         0.1834346425002680,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975362
    ])
    
    x0_nodes = (x_gl + 1.0) / 2.0
    idx = np.argsort(x0_nodes)
    x0_nodes = x0_nodes[idx]
    
    initial_node_guesses = [
        x0_nodes,  # Gauss-Legendre
        (np.arange(n) + 0.5)/n,  # Midpoint
        np.linspace(0.05, 0.95, n),  # Uniform
        np.sqrt(np.linspace(0.01, 0.99, n)),  # Sqrt-spaced
        1-np.sqrt(np.linspace(0.01, 0.99, n)),  # Inverse sqrt
        np.array([0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
        np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.99]),
    ]
    
    best_nodes = None
    best_score = np.inf
    
    for node_init in initial_node_guesses:
        # Ensure valid initial guess
        node_init = np.sort(node_init)
        if np.any(node_init <= 1e-6) or np.any(node_init >= 1-1e-6):
            continue
        if np.any(np.diff(node_init) <= 1e-4):
            continue
        
        try:
            result = minimize(objective_nodes, node_init, method='SLSQP',
                             bounds=[(1e-6, 1-1e-6)] * n,
                             options={'maxiter': 300, 'ftol': 1e-10, 'disp': False})
            
            if result.fun < best_score:
                best_score = result.fun
                best_nodes = result.x
        except Exception as e:
            pass
    
    if best_nodes is None:
        # Fallback to Gauss-Legendre
        nodes = (x_gl + 1.0) / 2.0
        weights = np.array([
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763
        ]) / 2.0
        idx = np.argsort(nodes)
        nodes = nodes[idx]
        weights = weights[idx]
    else:
        nodes = best_nodes
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        nodes = np.sort(nodes)
        
        # Find optimal weights for these nodes
        all_z = []
        all_w0 = []
        all_wa = []
        all_exact = []
        
        for w0, wa in cosmologies:
            for z in z_values:
                exact = exact_integral(z, w0, wa)
                if abs(exact) > 1e-12:
                    all_z.append(z)
                    all_w0.append(w0)
                    all_wa.append(wa)
                    all_exact.append(exact)
        
        m = len(all_z)
        
        A = np.zeros((m, n))
        b = np.zeros(m)
        
        for j in range(m):
            z_j = all_z[j]
            w0_j = all_w0[j]
            wa_j = all_wa[j]
            exact_j = all_exact[j]
            
            a_vals = 1.0/(1.0 + z_j * nodes)
            E_vals = E_of_a(a_vals, w0_j, wa_j)
            A[j, :] = z_j * nodes / E_vals
            b[j] = exact_j
        
        mask = np.abs(b) > 1e-12
        A_rel = A[mask, :] / b[mask, :][:, None]
        b_rel = np.ones(np.sum(mask))
        
        from scipy.optimize import minimize as min_opt
        
        def weight_objective(w):
            if np.any(w <= 0):
                return 1e10
            if abs(np.sum(w) - 1.0) > 1e-8:
                return 1e10
            residuals = A_rel @ w - b_rel
            return np.sum(residuals**2)
        
        w_init = np.ones(n) / n
        
        try:
            w_result = min_opt(weight_objective, w_init, method='SLSQP',
                              bounds=[(1e-10, 1.0)] * n,
                              constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
                              options={'maxiter': 500, 'ftol': 1e-12})
            
            if w_result.success and np.all(w_result.x > 0):
                weights = w_result.x
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n) / n
        except:
            weights = np.ones(n) / n
    
    # Final validation and cleanup
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    nodes = np.sort(nodes)
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
