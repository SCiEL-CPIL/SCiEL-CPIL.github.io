import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        
        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Phase 1: Global Exploration with Differential Evolution
        # Small population to save budget for refinement
        try:
            de_res = differential_evolution(
                counted_func, 
                bounds=[(-5, 5)] * self.dim, 
                maxiter=5, 
                popsize=5, 
                tol=1e-6,
                seed=0,
                init='latinhypercube'
            )
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x
        except Exception:
            pass

        # Phase 2: Local Refinement with Powell
        if time.time() < deadline and call_count < max_calls:
            try:
                nm_res = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell',
                    options={'xtol': 1e-10, 'ftol': 1e-10}
                )
                if nm_res.fun < best_f:
                    best_f = nm_res.fun
                    best_x = nm_res.x
            except Exception:
                pass

        return best_x, best_f
