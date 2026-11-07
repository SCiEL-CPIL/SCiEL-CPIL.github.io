"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time and budget guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(123)
        
        best_val = float('inf')
        best_x = np.zeros(dim)
        call_count = 0
        start_time = time.time()
        # Leave a safety margin for Python overhead and evaluator checks
        deadline = start_time + 290.0 

        class BudgetExceeded(Exception):
            pass

        def counting_func(x):
            nonlocal call_count, best_val, best_x
            call_count += 1
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            # Check limits
            if call_count >= budget:
                raise BudgetExceeded()
            if time.time() > deadline:
                raise BudgetExceeded()
                
            return val

        # Generate diverse starting points
        # 1. Origin (0,0,0,0,0) - often a good starting point for centered functions
        # 2. Stratified random points to cover the space
        starts = [np.zeros(dim)]
        
        # Create a grid of points or random points
        # Let's use 9 random points to ensure coverage
        for _ in range(9):
            starts.append(rng.uniform(-5.0, 5.0, dim))
            
        # Sort starts? Maybe put origin first.
        # We will run L-BFGS-B from each start.
        
        for x0 in starts:
            if call_count >= budget or time.time() > deadline:
                break
                
            try:
                # L-BFGS-B is efficient. 
                # We don't set a hard maxfun here because counting_func handles it,
                # but we set a high maxiter to let it run until budget/time is hit.
                minimize(counting_func, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000, 'maxfun': 1000})
            except BudgetExceeded:
                break
            except Exception:
                # If L-BFGS-B fails for other reasons, try the next start
                continue

        return best_x, best_val
# EVOLVE-BLOCK-END
