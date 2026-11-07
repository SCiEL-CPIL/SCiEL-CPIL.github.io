"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # Count function calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Initial random point
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4, 4, self.dim)
        best_x = x0.copy()
        best_f = counted_func(best_x)
        
        # Multi-start strategy: We have 500 calls.
        # Nelder-Mead in 5D typically uses ~50-100 calls to converge if close.
        # We can afford 3-4 restarts.
        # Let's try 4 restarts.
        num_restarts = 4
        calls_per_restart = (self.budget - call_count[0]) // num_restarts
        
        # If we are very tight on budget, reduce restarts
        if calls_per_restart < 20:
            num_restarts = 1
            calls_per_restart = self.budget - call_count[0]

        for i in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # New starting point
            if i == 0:
                x_start = best_x
            else:
                x_start = rng.uniform(-4, 4, self.dim)
                # Ensure we don't start at the exact same place
                # But we already evaluated x0. If x_start is close to best_x, 
                # it might be redundant, but NM handles it.
                
            # Run Nelder-Mead
            # maxiter is a proxy for calls, but NM call count is variable.
            # We use maxiter to limit the number of simplex iterations.
            # In 5D, each iteration is ~1-2 function evaluations.
            # So maxiter ~ calls_per_restart is a safe upper bound.
            
            try:
                res = minimize(
                    counted_func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': calls_per_restart, 
                        'fatol': 1e-10, 
                        'xatol': 1e-10
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                # If optimization fails, continue to next restart
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
