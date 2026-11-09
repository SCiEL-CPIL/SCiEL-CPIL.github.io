import numpy as np
import time
from scipy.optimize import minimize
from cma import CMAEvolutionStrategy

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 290.0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility if needed, but default is fine
        # CMA-ES handles the randomness internally
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (dim,):
                try:
                    x = np.reshape(x, (dim,))
                except:
                    return float('inf')
            
            self.call_count += 1
            try:
                f_val = float(func(x))
            except:
                return float('inf')
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            
            return f_val

        # Strategy: CMA-ES with a good initial point
        # CMA-ES is robust for multimodal functions if given enough budget.
        # We will use most of the budget for CMA-ES.
        
        # Initial point: Center of the box
        x0 = np.zeros(dim)
        
        # Initial step size: 20% of the box width (10.0) is a common heuristic
        sigma0 = 2.0
        
        # Configure CMA-ES
        # maxiter is not the budget limit, we control it via calls
        # popsize: default is 4 + floor(3 ln N), for N=5 that is 4+3*1.6=8. 
        # With 500 budget, we can afford a larger population for better exploration.
        # Let's try popsize 10 or 15 to improve multimodal handling.
        
        opts = {
            'maxfevals': budget,  # Hard limit on function evaluations
            'tolfun': 1e-12,
            'tolfunint': 1e-12,
            'tolfunup': 1e-12,
            'tolfunhist': 1e-12,
            'maxiter': 10000, # Large number, maxfevals will stop it
            'popsize': 12,    # Larger population for better global search in 5D
            'verbose': -9,    # Suppress output
            'seed': 42,
            'bounds': [(-5.0, 5.0)] * dim
        }
        
        # We need to be careful with the call count.
        # CMA-ES might try to evaluate more than budget if not careful, 
        # but maxfevals should handle it.
        # However, the wrapper checks budget strictly.
        
        try:
            es = CMAEvolutionStrategy(x0, sigma0, wrapped_func, opts)
            es.optimize()
        except Exception as e:
            # If CMA-ES fails for any reason, fallback to simple random search or return best so far
            pass
        
        # If we still have budget left and CMA-ES stopped early (e.g., due to tol),
        # we could do a local polish, but CMA-ES usually converges well.
        # Given the strict budget and time limit, we trust CMA-ES to use its budget efficiently.
        
        # Fallback if no calls were made (should not happen)
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            if self.call_count < budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
