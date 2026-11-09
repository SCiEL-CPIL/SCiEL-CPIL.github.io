"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
from scipy.stats import qmc

class TimeLimitExceeded(Exception):
    pass

class Optimizer:
    """Multi-start Nelder-Mead with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        # Safety margin: 10 seconds before the 300s limit
        self.deadline = self.start_time + 290.0

    def __call__(self, func):
        best_x = None
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal self
            if self.calls >= self.budget:
                raise TimeLimitExceeded("Budget exhausted")
            if time.time() > self.deadline:
                raise TimeLimitExceeded("Time limit exceeded")
            self.calls += 1
            return func(x)

        # Generate 5 diverse starting points using Sobol sequence in [-4, 4]^5
        # The minimizer is known to be in [-4, 4]^5
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(5)
        starts = points * 8.0 - 4.0
        
        # Ensure we have at least one point near the center, as it's a common default
        # But Sobol is already well-distributed. We'll stick with Sobol for consistency.
        
        for x0 in starts:
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break

            # Calculate max iterations for this run based on remaining budget
            # Nelder-Mead in 5D approx 6-8 function calls per iteration
            remaining_budget = self.budget - self.calls
            max_iter = max(1, remaining_budget // 7)
            
            # Cap max_iter to prevent a single run from dominating if others are short
            # But since we check budget inside, it's safe.
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-5, 
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except TimeLimitExceeded:
                break
            except Exception:
                # If any other error occurs, break to ensure we return something
                break

        # If no valid point was found (e.g., all exceptions), return a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x) if self.calls < self.budget else float('inf')

        return best_x, best_f
# EVOLVE-BLOCK-END
