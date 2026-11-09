import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.max_calls = self.budget
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf
        
        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            if self.calls >= self.max_calls:
                return self.best_f
            self.calls += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Multi-start points
        # 1. Origin
        # 2. Random points
        np.random.seed(42)
        starts = [np.zeros(self.dim)]
        
        # Add random starts
        # We have 500 budget. Let's try 5 starts.
        # Nelder-Mead in 5D uses 6 points in simplex. 
        # Each iteration uses ~1-6 function calls.
        # 100 calls per start is reasonable.
        
        num_starts = 5
        for _ in range(num_starts - 1):
            starts.append(np.random.uniform(-5, 5, self.dim))
        
        # Allocate budget per start
        calls_per_start = self.budget // num_starts
        
        for x0 in starts:
            if self.calls >= self.max_calls:
                break
            
            # Clip x0 to bounds
            x0 = np.clip(x0, -5, 5)
            
            # Run Nelder-Mead
            # maxiter is not strictly function calls, so we rely on the wrapper 
            # stopping if budget is hit, but we should also limit iterations to be safe.
            # A typical NM iteration might use 1-5 evals. 
            # Let's set maxiter to calls_per_start to be conservative.
            try:
                result = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': calls_per_start, 'xatol': 1e-4, 'fatol': 1e-4})
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = result.x
            except Exception:
                pass
                
        return self.best_x, self.best_f
