"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * n
        
        # State for tracking best solution and call count
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]
        
        def wrapped_func(x):
            call_count[0] += 1
            val = func(x)
            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x.copy()
            return val
            
        def callback(xk):
            # Stop if we are getting close to the budget limit
            if call_count[0] >= budget - 10:
                return True
            return None

        # Generate starting points
        rng = np.random.default_rng(123)
        starts = [np.zeros(n)]
        # Add random starts to escape local minima
        for _ in range(5):
            starts.append(rng.uniform(-5.0, 5.0, n))
            
        for start in starts:
            if call_count[0] >= budget:
                break
                
            try:
                res = minimize(wrapped_func, start, method='L-BFGS-B', 
                               bounds=bounds, callback=callback, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
            except Exception:
                pass

        # If no valid result, return the first evaluated point or zero
        if best_x[0] is None:
            best_x[0] = np.zeros(n)
            best_f[0] = wrapped_func(best_x[0])

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
