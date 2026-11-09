import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def __call__(self, func):
        self.call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return 1e10
            self.call_count += 1
            x = np.asarray(x, dtype=float)
            val = float(func(x))
            if val < best_f:
                nonlocal best_f
                best_f = val
            return val

        # Initial evaluation
        best_f = wrapped_func(best_x)
        
        # Use a local random generator for shuffling dimensions
        rng = np.random.default_rng(42)
        
        # Perform multiple cycles of cyclic coordinate descent
        # Each cycle optimizes each dimension once.
        # We run until budget is nearly exhausted.
        
        # Estimate calls per cycle:
        # minimize_scalar bounded typically uses ~10-20 calls.
        # 5 dims * 20 calls = 100 calls/cycle.
        # 500 calls / 100 = 5 cycles.
        
        # To be safe, we loop and break on budget.
        
        for cycle in range(20): # Max 20 cycles
            if self.call_count >= self.budget - 10: # Leave buffer
                break
                
            # Shuffle dimension order to improve convergence properties
            dims = list(range(self.dim))
            rng.shuffle(dims)
            
            for i in dims:
                if self.call_count >= self.budget - 5:
                    break
                
                # Define 1D objective
                def func_1d(t):
                    x_temp = best_x.copy()
                    x_temp[i] = t
                    return wrapped_func(x_temp)
                
                try:
                    # Bounded minimization
                    # xatol=1e-4 is a good balance between precision and cost
                    res = minimize_scalar(
                        func_1d, 
                        bounds=(-5.0, 5.0), 
                        method='bounded',
                        options={'xatol': 1e-4}
                    )
                    if res.fun < best_f:
                        best_x[i] = res.x
                except Exception:
                    pass

        return best_x, best_f
