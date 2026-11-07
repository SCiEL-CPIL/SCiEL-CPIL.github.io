"""Convex optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    n = N
    # We use a random initialization near the boundary (0 or 1) to help escape the
    # local minimum at h=0.5, which is a saddle point/local max for the correlation.
    # We generate a random vector, then project it to the sum constraint.
    np.random.seed(42) # Fixed seed for reproducibility in evaluation context if needed, 
                       # but task says "seed is 100 heights of 0.5" for the START, 
                       # not necessarily for RNG. However, deterministic is safer.
    
    # Start with a random configuration biased towards 0 and 1
    # This helps the optimizer find a lower max correlation.
    h_init = np.random.rand(n)
    # Project to sum N/2
    h_init = h_init - np.mean(h_init) + 0.5
    h_init = np.clip(h_init, 0.0, 1.0)
    # Re-adjust sum precisely
    diff = (n / 2.0 - np.sum(h_init)) / n
    h_init = np.clip(h_init + diff, 0.0, 1.0)
    
    # Initial t should be an upper bound on the max correlation
    lags_init = np.correlate(h_init, 1 - h_init, mode="full")
    t_init = np.max(lags_init) + 0.1
    
    x0 = np.concatenate([np.array([t_init]), h_init])
    
    bounds = [(0, None)] + [(0, 1)] * n
    
    def objective(x):
        return x[0]
    
    def objective_grad(x):
        grad = np.zeros(n + 1)
        grad[0] = 1.0
        return grad
    
    def sum_con(x):
        return np.sum(x[1:]) - n / 2.0
    
    def sum_con_grad(x):
        g = np.zeros(n + 1)
        g[1:] = 1.0
        return g
    
    def lower_bounds_con(x):
        return x[1:]
    
    def lower_bounds_con_grad(x):
        g = np.zeros((n, n + 1))
        g[np.arange(n), 1 + np.arange(n)] = 1.0
        return g
    
    def upper_bounds_con(x):
        return 1.0 - x[1:]
    
    def upper_bounds_con_grad(x):
        g = np.zeros((n, n + 1))
        g[np.arange(n), 1 + np.arange(n)] = -1.0
        return g
    
    def corr_con(x):
        t = x[0]
        h = x[1:]
        lags = np.correlate(h, 1 - h, mode="full")
        return t - lags

    # We provide analytical gradients for the sum and bounds constraints.
    # For correlation constraints, we let SLSQP use numerical gradients to avoid
    # complex Jacobian construction, which is acceptable for N=100.
    
    constraints = [
        {'type': 'eq', 'fun': sum_con, 'jac': sum_con_grad},
        {'type': 'ineq', 'fun': lower_bounds_con, 'jac': lower_bounds_con_grad},
        {'type': 'ineq', 'fun': upper_bounds_con, 'jac': upper_bounds_con_grad},
        {'type': 'ineq', 'fun': corr_con},
    ]
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            h = result.x[1:]
            # Post-processing to ensure strict validity
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if abs(current_sum - n / 2.0) > 1e-9 * n:
                diff = (n / 2.0 - current_sum) / n
                h = h + diff
                h = np.clip(h, 0.0, 1.0)
            
            # Final check
            if not np.all(np.isfinite(h)):
                return np.full(n, 0.5)
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return np.full(n, 0.5)
            if abs(np.sum(h) - n / 2.0) > 1e-9 * n:
                return np.full(n, 0.5)
                
            return h
        else:
            # Fallback: use the random initialization if optimization fails
            # This is likely better than the 0.5 seed
            h = h_init
            # Ensure validity of fallback
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            diff = (n / 2.0 - current_sum) / n
            h = np.clip(h + diff, 0.0, 1.0)
            return h
    except Exception:
        return np.full(n, 0.5)
# EVOLVE-BLOCK-END
