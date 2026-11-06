"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linprog, differential_evolution
from scipy.integrate import quad

def construct():
    """
    Construct an 8-node positive quadrature rule using a combined approach:
    1. Fix nodes to Gauss-Legendre nodes on [0,1] (optimal for smooth functions)
    2. Formulate weight optimization as a linear programming problem to minimize
       worst-case error over a dense set of representative integrands
    3. If LP fails, fall back to Nelder-Mead on both nodes and weights
    """
    n = 8
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    def time_remaining():
        return deadline - time.time()
    
    # E(z) for flat CPL model
    def E_z(z, Omega_m, w0, wa):
        """H(z)/H0 for flat CPL model"""
        Omega_de = 1.0 - Omega_m
        z = np.asarray(z, dtype=float)
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_de * (1.0 + z)**(3.0*(1.0 + w0 + wa*z))
        return np.sqrt(term_m + term_de)
    
    # The integrand for the quadrature: h(t) = 1/E(1/(1+z*t))
    # We're approximating: integral_0^1 1/E(1/(1+z*t)) dt
    def integrand_h(t, z, Omega_m, w0, wa):
        """h(t) = 1/E(1/(1+z*t)) where E is evaluated at redshift z*t"""
        zt = z * t
        return 1.0 / E_z(zt, Omega_m, w0, wa)
    
    # Generate representative test cases
    # Cosmologies: (Omega_m, w0, wa)
    test_cosmologies = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.20, -1.0, 0.0),
        (0.40, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.1, 0.0),
        (0.30, -1.0, 1.0),
        (0.30, -1.0, -1.0),
        (0.25, -1.0, 0.3),
        (0.35, -1.0, -0.3),
    ]
    
    # Test redshifts: log-spaced from 0.01 to 10
    z_test_values = np.logspace(-2, 1, 12)
    
    # Pre-compute exact integrals for all test cases
    def compute_exact_integrals():
        """Pre-compute exact integrals for all test cases"""
        exact_integrals = []
        for Om, w0, wa in test_cosmologies:
            for z in z_test_values:
                if time_remaining() < 5.0:
                    break
                f = lambda t: integrand_h(t, z, Om, w0, wa)
                exact_val, _ = quad(f, 0, 1, limit=100, epsabs=1e-10, epsrel=1e-10)
                exact_integrals.append(exact_val)
        return np.array(exact_integrals)
    
    # Strategy 1: Fix nodes to Gauss-Legendre, optimize weights via LP
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    gl_nodes = (x_gl + 1.0) / 2.0  # Map to [0,1]
    gl_weights = w_gl / 2.0
    gl_weights = gl_weights / np.sum(gl_weights)
    
    def solve_lp_weights():
        """
        Solve LP: minimize max_i |sum_j w_j * f_i(x_j) - I_i| / I_i
        Subject to: w_j > 0, sum_j w_j = 1
        
        This is a linear program:
        Minimize: M
        Subject to:
            M >= (sum_j w_j * f_i(x_j) - I_i) / I_i  for all i
            M >= -(sum_j w_j * f_i(x_j) - I_i) / I_i  for all i
            sum_j w_j = 1
            w_j >= 0
        """
        if time_remaining() < 10.0:
            return None
        
        # Compute f_i(x_j) for all test cases i and nodes j
        num_cases = len(test_cosmologies) * len(z_test_values)
        F = np.zeros((num_cases, n))
        I = np.zeros(num_cases)
        
        idx = 0
        for Om, w0, wa in test_cosmologies:
            for z in z_test_values:
                if time_remaining() < 5.0:
                    return None
                for j in range(n):
                    F[idx, j] = integrand_h(gl_nodes[j], z, Om, w0, wa)
                f = lambda t: integrand_h(t, z, Om, w0, wa)
                I[idx], _ = quad(f, 0, 1, limit=100, epsabs=1e-10, epsrel=1e-10)
                idx += 1
        
        # Filter out cases with very small I
        valid = I > 1e-10
        F = F[valid]
        I = I[valid]
        num_valid = len(I)
        
        if num_valid == 0:
            return None
        
        # Normalize: divide by I to get relative error
        F_norm = F / I  # F_norm[i, j] = f_i(x_j) / I_i
        
        # LP formulation:
        # Variables: w_0, ..., w_{n-1}, M
        # Minimize: M
        # Subject to:
        #   sum_j w_j * F_norm[i, j] - 1 <= M  for all i
        #   1 - sum_j w_j * F_norm[i, j] <= M  for all i
        #   sum_j w_j = 1
        #   w_j >= 0
        #   M >= 0
        
        # c: objective coefficients [0, ..., 0, 1] (last is M)
        c = np.zeros(n + 1)
        c[-1] = 1.0
        
        # Inequality constraints: A_ub @ x <= b_ub
        # For each i: sum_j w_j * F_norm[i,j] - M <= 1
        #             -sum_j w_j * F_norm[i,j] - M <= -1
        A_ub = np.zeros((2 * num_valid, n + 1))
        b_ub = np.zeros(2 * num_valid)
        
        for i in range(num_valid):
            # Row 2*i: sum_j w_j * F_norm[i,j] - M <= 1
            A_ub[2*i, :n] = F_norm[i, :]
            A_ub[2*i, n] = -1.0
            b_ub[2*i] = 1.0
            
            # Row 2*i+1: -sum_j w_j * F_norm[i,j] - M <= -1
            A_ub[2*i+1, :n] = -F_norm[i, :]
            A_ub[2*i+1, n] = -1.0
            b_ub[2*i+1] = -1.0
        
        # Equality constraints: A_eq @ x = b_eq
        # sum_j w_j = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_j >= 0, M >= 0
        bounds = [(0, None)] * n + [(0, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs', options={'presolve': True})
            if result.success:
                weights = result.x[:n]
                # Normalize to ensure sum = 1 exactly
                weights = weights / np.sum(weights)
                if np.all(weights > 0):
                    return weights
        except Exception:
            pass
        
        return None
    
    # Try LP approach
    lp_weights = solve_lp_weights()
    
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    best_error = np.inf
    
    # Evaluate error for a given (nodes, weights)
    def evaluate_worst_error(nodes, weights):
        """Compute worst-case relative error over test cases"""
        max_err = 0.0
        for Om, w0, wa in test_cosmologies:
            for z in z_test_values:
                if time_remaining() < 5.0:
                    break
                # Quadrature approximation
                quad_val = np.sum(weights * integrand_h(nodes, z, Om, w0, wa))
                
                # Exact integral
                f = lambda t: integrand_h(t, z, Om, w0, wa)
                exact_val, _ = quad(f, 0, 1, limit=100, epsabs=1e-10, epsrel=1e-10)
                
                if abs(exact_val) > 1e-10:
                    err = abs(quad_val - exact_val) / abs(exact_val)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # If LP succeeded, evaluate it
    if lp_weights is not None:
        lp_error = evaluate_worst_error(gl_nodes, lp_weights)
        if lp_error < best_error:
            best_nodes = gl_nodes.copy()
            best_weights = lp_weights.copy()
            best_error = lp_error
    
    # If LP didn't help enough, try Nelder-Mead on both nodes and weights
    if time_remaining() > 15.0:
        def objective(params):
            """Objective: worst-case error with transformed parameters"""
            if time_remaining() < 5.0:
                return np.inf
            
            node_logits = params[:n]
            weight_logs = params[n:2*n]
            
            nodes = 1.0 / (1.0 + np.exp(-node_logits))
            sort_idx = np.argsort(nodes)
            nodes = nodes[sort_idx]
            weight_logs = weight_logs[sort_idx]
            
            if np.any(np.diff(nodes) <= 1e-10):
                return np.inf
            
            weights = np.exp(weight_logs)
            weights = weights / np.sum(weights)
            
            return evaluate_worst_error(nodes, weights) * 100.0
        
        # Start from GL nodes and best weights found so far
        init_nodes = best_nodes
        init_weights = best_weights
        
        init_node_logits = np.log(init_nodes / (1.0 - init_nodes))
        init_weight_logs = np.log(init_weights)
        init_params = np.concatenate([init_node_logits, init_weight_logs])
        
        from scipy.optimize import minimize
        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'fatol': 1e-8, 'adaptive': True, 'xatol': 1e-8}
            )
            
            if result.fun < best_error * 100.0:
                node_logits = result.x[:n]
                weight_logs = result.x[n:2*n]
                
                nodes = 1.0 / (1.0 + np.exp(-node_logits))
                sort_idx = np.argsort(nodes)
                nodes = nodes[sort_idx]
                weight_logs = weight_logs[sort_idx]
                
                weights = np.exp(weight_logs)
                weights = weights / np.sum(weights)
                
                if np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 1e-10) and np.all(weights > 0):
                    best_nodes = nodes
                    best_weights = weights
                    best_error = result.fun / 100.0
        except Exception:
            pass
    
    # Final validation
    nodes = best_nodes
    weights = best_weights
    
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 1e-10)):
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        weights = weights / np.sum(weights)
    
    if not np.all(weights > 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
