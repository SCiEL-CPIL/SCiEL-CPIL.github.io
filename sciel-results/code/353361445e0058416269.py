import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280
        
        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0
        
        current_best_x = np.zeros(n)
        current_best_f = float('inf')

        def wrapper(x):
            nonlocal calls, current_best_x, current_best_f
            if calls >= budget or time.time() > deadline:
                raise ValueError("Budget or time limit exceeded")
            calls += 1
            f_val = func(x)
            if f_val < current_best_f:
                current_best_f = f_val
                current_best_x = x.copy()
            return f_val

        # Deterministic starting points
        starts = [
            np.zeros(n),
            np.ones(n) * 1.0,
            np.ones(n) * -1.0,
            np.ones(n) * 3.0,
            np.ones(n) * -3.0,
        ]
        
        rng = np.random.RandomState(42)
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, size=n))

        for start in starts:
            if calls >= budget or time.time() > deadline:
                break
            
            current_best_x = start.copy()
            current_best_f = float('inf')
            
            try:
                res = minimize(wrapper, start, method='Nelder-Mead', 
                               options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'initial_simplex': 2.0})
            except Exception:
                pass
            
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = current_best_x

        return best_x, best_f
