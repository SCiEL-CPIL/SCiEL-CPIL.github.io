"""
Strategy: Optimize the 8 nodes and 8 weights to minimize the worst-case relative
error across a representative set of CPL cosmologies and redshifts.
This treats the quadrature rule as a parameterized object to be tuned specifically
for the class of integrands 1/E(a; w0, wa) rather than relying on a general-purpose
spectral rule like Gauss-Legendre.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds (leave 5s margin for 90s limit)

def construct():
    """
    Construct the 8-node positive quadrature rule.
    Returns:
        dict with 'nodes' (length-8) and 'weights' (length-8).
    """
    
    # 1. Define the CPL E(z) function
    def E_z(z, w0, wa):
        if z < 0:
            return 1.0
        term1 = (1 + z) ** (-3 * w0)
        base = 1 - wa * z / (1 + z)
        if base < 0:
            base = 1e-12
        term2 = base ** (1.5 * wa)
        return np.sqrt(term1 * term2)
    
    # 2. Define the true integral using high-resolution Simpson's rule (vectorized)
    # The integral is I = \int_0^1 dx / E(z*x)
    # We use a fixed high-resolution grid for the "true" value during optimization
    # to ensure consistency and speed.
    N_TRUE = 2000
    x_true = np.linspace(0, 1, N_TRUE)
    
    def true_integral(z, w0, wa):
        if z == 0:
            return 1.0
        z_vals = z * x_true
        E_vals = E_z(z_vals, w0, wa)
        integrand = 1.0 / E_vals
        # Simpson's rule requires odd number of points (even number of intervals)
        # N_TRUE=2000 is even, so N_TRUE-1 intervals is odd? No.
        # Simpson's 1/3 rule: n intervals must be even.
        # Let's use N_TRUE=2001 points -> 2000 intervals (even).
        # Wait, np.linspace(0,1,2001) gives 2001 points.
        # Let's stick to a large N and use trapz for simplicity and robustness, 
        # or ensure Simpson is correct.
        # Actually, for optimization, relative error is what matters.
        # Let's use a very accurate quadrature for the "true" value.
        # scipy.integrate.quad is too slow if called many times.
        # Let's use a high-order Gauss-Legendre for the "true" reference.
        # This is safe because we are not evaluating the target rule, but the reference.
        nodes_ref, weights_ref = np.polynomial.legendre.leggauss(64)
        x_ref = 0.5 * (nodes_ref + 1.0)
        w_ref = 0.5 * weights_ref
        
        z_vals_ref = z * x_ref
        E_vals_ref = E_z(z_vals_ref, w0, wa)
        integrand_ref = 1.0 / E_vals_ref
        return np.sum(w_ref * integrand_ref)

    # 3. Define the quadrature approximation
    def quadrature_approx(nodes, weights, z, w0, wa):
        if z == 0:
            return 1.0
        z_vals = z * nodes
        E_vals = E_z(z_vals, w0, wa)
        integrand = 1.0 / E_vals
        return np.sum(weights * integrand)

    # 4. Define the objective function for optimization
    # We want to minimize the max relative error over a set of test cases.
    # Test cases: 
    #   z in [0.5, 1.0, 2.0, 5.0]
    #   w0 in [-1.5, -1.0, -0.5, 0.0, 0.5]
    #   wa in [-0.5, 0.0, 0.5, 1.0]
    
    z_test = [0.5, 1.0, 2.0, 5.0]
    w0_test = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_test = [-0.5, 0.0, 0.5, 1.0]
    
    # Pre-compute true integrals for all test cases to save time in objective
    # There are 4 * 5 * 4 = 80 cases.
    # This is fast enough.
    
    def objective(params):
        # params: [nodes (8), weights (8)]
        # Constraints:
        # 1. nodes strictly increasing in [0,1]
        # 2. weights > 0
        # 3. sum(weights) = 1
        
        # We will enforce constraints by transforming params or using penalty.
        # L-BFGS-B supports bounds. 
        # To enforce sum(weights)=1 and nodes ordering, it's tricky with simple bounds.
        # Alternative: Optimize unconstrained params and map them?
        # Or use penalty method in objective.
        
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e6
        if np.any(nodes < 0) or np.any(nodes > 1):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e6 + abs(np.sum(weights) - 1.0)
            
        max_err = 0.0
        for z in z_test:
            for w0 in w0_test:
                for wa in wa_test:
                    I_true = true_integral(z, w0, wa)
                    I_approx = quadrature_approx(nodes, weights, z, w0, wa)
                    rel_err = abs(I_approx - I_true) / I_true
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err

    # 5. Initial guess
    # Start with Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (nodes_gl + 1.0)
    weights_init = 0.5 * weights_gl
    x0 = np.concatenate([nodes_init, weights_init])
    
    # 6. Bounds
    # Nodes: (0, 1)
    # Weights: (0, 1)
    # We need to handle the ordering constraint. 
    # A common trick: optimize log-spaced nodes or use a transformation.
    # However, for 8 nodes, we can just use penalty in objective and hope L-BFGS-B stays in bounds.
    # Better: Define bounds for nodes as [1e-6, 1-1e-6] and weights [1e-6, 1].
    # The ordering constraint is the hard part.
    # Let's try a different parameterization:
    # Let nodes be defined by cumulative sums of positive variables.
    # n_i = sum_{j=1}^i d_j, where d_j > 0 and sum d_j = 1.
    # This ensures 0 < n_1 < ... < n_8 < 1.
    # Let d_j = exp(y_j) / sum(exp(y_k)).
    # Then nodes are determined by y (8 variables).
    # Weights: w_i = exp(v_i) / sum(exp(v_k)).
    # Total params: 8 (y) + 8 (v) = 16.
    # Objective takes y, v, converts to nodes, weights, and evaluates error.
    
    def objective_transformed(params):
        y = params[:8]
        v = params[8:]
        
        # Softmax for differences d
        d_unnorm = np.exp(y)
        d = d_unnorm / np.sum(d_unnorm)
        nodes = np.cumsum(d)
        # nodes[7] should be 1.0. But we want nodes < 1.
        # The mapping above makes nodes[7]=1.
        # We want nodes in (0,1).
        # Let's scale: nodes = nodes * (1 - 2*eps) + eps?
        # Or just accept nodes[7] close to 1.
        # Actually, the integral is from 0 to 1. The nodes should be inside.
        # If we use cumulative sum of positive numbers, the last node is 1.
        # We need to shrink them into (0,1).
        # Let's use: nodes = eps + (1-2*eps) * (np.cumsum(d) / np.sum(d))
        # Wait, np.cumsum(d) already sums to 1.
        # So nodes = eps + (1-2*eps) * np.cumsum(d)
        
        eps = 1e-6
        nodes = eps + (1 - 2 * eps) * np.cumsum(d)
        
        # Softmax for weights
        w_unnorm = np.exp(v)
        weights = w_unnorm / np.sum(w_unnorm)
        
        # Evaluate error
        max_err = 0.0
        for z in z_test:
            for w0 in w0_test:
                for wa in wa_test:
                    I_true = true_integral(z, w0, wa)
                    I_approx = quadrature_approx(nodes, weights, z, w0, wa)
                    rel_err = abs(I_approx - I_true) / I_true
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err

    # Initial params for transformed space
    # Inverse of nodes -> y
    # d = diff(nodes)
    # y = log(d)
    d_init = np.diff(np.concatenate([[0], nodes_init, [1]]))
    d_init = d_init[1:7] # Wait, nodes_init has 8 points.
    # The transformation: nodes = eps + (1-2eps) * cumsum(d)
    # So cumsum(d) = (nodes - eps) / (1-2eps)
    # d = diff(cumsum(d))
    c_init = (nodes_init - eps) / (1 - 2 * eps)
    d_init = np.diff(np.concatenate([[0], c_init]))
    y_init = np.log(d_init)
    
    # Weights
    v_init = np.log(weights_init)
    
    x0_trans = np.concatenate([y_init, v_init])
    
    # 7. Optimize
    best_result = None
    best_score = float('inf')
    
    try:
        # Use L-BFGS-B
        result = minimize(
            objective_transformed,
            x0_trans,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_score:
            best_result = result
            best_score = result.fun
            
        # Try a few random restarts if time permits
        if time.time() < DEADLINE - 10:
            for i in range(5):
                # Perturb initial guess
                noise = np.random.normal(0, 0.1, size=16)
                x0_pert = x0_trans + noise
                result = minimize(
                    objective_transformed,
                    x0_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
                if time.time() > DEADLINE - 5:
                    break
                    
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        # Fallback to GL
        return {"nodes": nodes_init, "weights": weights_init}

    if best_result is not None:
        y_opt = best_result.x[:8]
        v_opt = best_result.x[8:]
        
        d_unnorm = np.exp(y_opt)
        d = d_unnorm / np.sum(d_unnorm)
        c_opt = np.cumsum(d)
        nodes_opt = eps + (1 - 2 * eps) * c_opt
        
        w_unnorm = np.exp(v_opt)
        weights_opt = w_unnorm / np.sum(w_unnorm)
        
        # Final sanity check
        if np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and np.all(np.diff(nodes_opt) > 0) and np.all(weights_opt > 0) and abs(np.sum(weights_opt) - 1) < 1e-10:
            return {"nodes": nodes_opt, "weights": weights_opt}
            
    # Fallback
    return {"nodes": nodes_init, "weights": weights_init}

# EVOLVE-BLOCK-END
