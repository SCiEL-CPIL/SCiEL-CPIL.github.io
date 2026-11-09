"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Generate diverse starting points
        # We'll use a mix of random and structured points
        rng = np.random.default_rng(42)
        
        # Number of starts
        num_starts = 10
        calls_per_start = max(1, self.budget // num_starts)
        
        best_x = None
        best_f = np.inf
        
        # Initial candidates
        # 1. Origin (0,0,0,0,0) - often good for Rosenbrock if shift is small
        # 2. Random points
        # 3. Points near boundaries
        
        candidates = []
        candidates.append(np.zeros(self.dim))
        
        # Add random points
        for _ in range(num_starts - 1):
            candidates.append(rng.uniform(-5, 5, self.dim))
        
        # Ensure we don't exceed budget in total
        # We will run minimize for each candidate
        
        for i, x0 in enumerate(candidates):
            if time.time() > deadline:
                break
                
            # Clip x0 to box
            x0 = np.clip(x0, -5, 5)
            
            # Define a wrapper to count calls and enforce deadline?
            # scipy minimize handles maxfev.
            # We need to be careful: if one run uses all calls, others get 0.
            # Better to strictly limit maxfev per run.
            
            try:
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': calls_per_start, 
                        'maxfev': calls_per_start,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                
                if res.success or res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass
        
        # If we have budget left and time, we could refine the best_x further
        # But Nelder-Mead is usually sufficient.
        
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
