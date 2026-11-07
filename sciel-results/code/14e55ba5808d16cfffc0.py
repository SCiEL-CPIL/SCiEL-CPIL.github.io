import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        start_time = time.time()
        
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() - start_time > 280:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global search with Multi-Start Nelder-Mead
        starts = [np.zeros(dim)]
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        for start in starts:
            if calls >= budget or time.time() - start_time > 280:
                break
            
            remaining = budget - calls
            max_iter = max(10, remaining // 3) # Use 1/3 of remaining for each start
            
            try:
                minimize(wrapped_func, start, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        # Phase 2: Local polish
        if calls < budget and time.time() - start_time < 280:
            remaining = budget - calls
            max_iter = max(10, remaining)
            try:
                minimize(wrapped_func, best_x, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                pass

        return best_x, best_f
