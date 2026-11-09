"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-Start L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def _counted_func(self, func):
        """Wrapper to count function evaluations and enforce budget."""
        def wrapper(x):
            if self.eval_count >= self.budget:
                # Return a very high value to stop optimization
                return 1e10
            self.eval_count += 1
            # Safety margin: if we are close to limit, stop
            if self.eval_count > self.budget - 5:
                return 1e10
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val
        return wrapper

    def __call__(self, func):
        self.eval_count = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        self.start_time = time.time()
        
        cf = self._counted_func(func)
        rng = np.random.default_rng(42)
        
        # Generate starting points
        # 1. Center
        starts = [np.zeros(self.dim)]
        # 2. Random points
        # We want to cover the space. 
        # Let's use 4 random points.
        for _ in range(4):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # We have 500 budget.
        # Let's allocate budget to each start.
        # Strategy: 
        # - Give the first start (center) a larger budget to see if it's good.
        # - If it finds a very good solution, we might not need others.
        # - But to be safe against local minima, we should try others.
        
        # Dynamic allocation:
        # Start with center. Allow 200 evals.
        # Then try randoms with 60 evals each.
        
        allocations = [200, 60, 60, 60, 60]
        
        for i, x0 in enumerate(starts):
            if self.eval_count >= self.budget:
                break
            
            # Check time safety
            if time.time() - self.start_time > 250:
                break
                
            maxfun = allocations[i] if i < len(allocations) else 50
            
            try:
                res = minimize(cf, x0, method='L-BFGS-B', bounds=self.bounds, 
                               options={'maxiter': 100, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
