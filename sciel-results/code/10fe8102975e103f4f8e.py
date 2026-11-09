from scipy.optimize import minimize
import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return best result before 630s limit
    start_time = time.time()
    deadline = start_time + 600  # 30s safety margin
    
    def laguerre_polynomial(x, m, n):
        """Compute Laguerre polynomial L_m^(n)(x)"""
        if m == 0:
            return np.ones_like(x)
        elif m == 1:
            return -(x + n)
        else:
            L_prev = np.ones_like(x)
            L_curr = -(x + n)
            for k in range(2, m + 1):
                L_new = ((2*k - 1 + n - x) * L_curr - (k - 1 + n) * L_prev) / k
                L_prev, L_curr = L_curr, L_new
            return L_curr
    
    def compute_bound(roots):
        """
        Estimate the Cohn-Goncalves upper bound.
        Based on the task description 'computes the last sign change',
        we assume the bound is proportional to the location of the last root/sign change.
        Minimizing the largest root minimizes the bound.
        """
        # The 'last sign change' for a polynomial with positive roots is typically
        # associated with the largest root.
        return np.max(roots)
    
    # Initial guess: Spread out roots to allow optimizer to converge
    # We start with a reasonable spread, but the optimizer will compress them.
    x0 = np.array([1.0 * (i + 1) for i in range(12)])
    
    # Objective function to minimize
    def objective(roots):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Timeout penalty
        
        # Ensure roots are positive and ordered
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            return 1e10
        
        return compute_bound(roots)
    
    # Constraints: roots must be positive and strictly increasing
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 1e-6},
        {'type': 'ineq', 'fun': lambda x: x[1:] - x[:-1] - 1e-6}
    ]
    
    # Bounds: each root between 1e-6 and 100
    bounds = [(1e-6, 100)] * 12
    
    # Try SLSQP optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if result.success and np.all(result.x > 0) and np.all(np.diff(result.x) > 0):
            return list(result.x)
        else:
            # Fall back to initial guess if optimization fails
            return list(x0)
            
    except Exception as e:
        # If optimization fails, return the initial arithmetic progression
        return list(x0)
    
# EVOLVE-BLOCK-END
