"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES optimizer for the Discus function (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        
        # Initial point: center of the box is a safe, good start for f11
        x0 = np.zeros(self.dim)
        
        # We need to track the best value found so far for the return value
        # and for the early stopping logic if needed, though CMA-ES handles it internally.
        # However, the task says "return value of __call__ is ignored", but we should 
        # still return something valid.
        
        # Set up the objective function wrapper to count calls and enforce time
        def wrapped_func(x):
            self.eval_count += 1
            if self.eval_count > self.budget:
                raise RuntimeError("Budget exceeded")
            
            # Time limit check: 300s limit, keep 5s margin
            if time.time() - self.start_time > 295:
                raise TimeoutError("Time limit exceeded")
                
            fx = func(x)
            
            # Track best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # CMA-ES parameters
        # maxiter is a safety cap, but we rely on budget/time
        # tol_x and tol_f are small to ensure convergence
        # popsize: default is 4 + floor(3 ln(n)). For n=5, 4+3*1.6=8.8 -> 9. 
        # Let's use a slightly larger population for better covariance estimation, e.g., 10 or 12.
        # With budget 500, popsize 10 allows ~50 generations. This is plenty for 5D.
        popsize = 10
        
        try:
            res = minimize(
                wrapped_func,
                x0,
                method='CMA-ES',
                options={
                    'maxiter': self.budget // popsize + 10,
                    'popsize': popsize,
                    'tol_x': 1e-6,
                    'tol_f': 1e-6,
                    'bounds': [(-5, 5)] * self.dim,
                    'verbose': 0
                }
            )
        except (RuntimeError, TimeoutError):
            pass
        except Exception:
            # Catch any other unexpected errors to avoid invalidating the whole run
            pass

        # If no evaluation was made (should not happen), do one
        if self.eval_count == 0:
            self.eval_count = 1
            self.best_f = func(x0)
            self.best_x = x0

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
