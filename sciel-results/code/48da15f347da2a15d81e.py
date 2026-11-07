"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        best_x = None
        best_fx = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Define a wrapper to count function calls
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_fx  # Return current best to avoid infinite loops
            calls_made += 1
            return func(x)

        # Multi-start Nelder-Mead
        # We'll use a few random starting points, but keep it simple.
        # For dim=5, NM is fast.
        
        # Start with a center point
        x0 = np.zeros(dim)
        res = minimize(counted_func, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_fx:
            best_fx = res.fun
            best_x = res.x
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_fx

        # Try a few random starts if budget allows
        # But careful not to exceed budget.
        # Each NM run takes ~100-200 calls typically for 5D.
        # Let's try 2 more random starts.
        
        for _ in range(2):
            if calls_made >= budget - 50 or time.time() > deadline:
                break
            x_start = rng.uniform(-5, 5, dim)
            res = minimize(counted_func, x_start, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_fx:
                best_fx = res.fun
                best_x = res.x

        return best_x, best_fx
# EVOLVE-BLOCK-END
