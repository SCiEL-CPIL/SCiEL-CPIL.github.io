import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e9
            if calls >= budget:
                return 1e9
            calls += 1
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start Nelder-Mead strategy
        # 1. Random initialization
        n_starts = 10
        max_iter_per_start = 50
        
        for i in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Choose start point
            if i == 0:
                # First start: center
                x0 = np.zeros(dim)
            else:
                # Subsequent starts: random points, possibly biased towards previous best
                if best_x is not None and np.random.rand() < 0.5:
                    # Perturb best
                    x0 = best_x + np.random.uniform(-1, 1, dim)
                else:
                    # Random
                    x0 = np.random.uniform(-5, 5, dim)
            
            x0 = np.clip(x0, -5, 5)
            
            # Local search
            try:
                res = minimize(wrapper, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_per_start, 'xatol': 1e-8, 'fatol': 1e-10})
            except:
                pass
            
            # Check if we improved
            if best_f is not None and res.fun < best_f:
                # Nelder-Mead might have used many calls, check budget
                pass

        return best_x, best_f
