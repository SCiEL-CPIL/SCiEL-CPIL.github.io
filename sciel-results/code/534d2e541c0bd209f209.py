import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 290.0  # Safety margin
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        self.current_calls = 0

        def counted_func(x):
            nonlocal best_x, best_f
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            self.current_calls += 1
            if self.current_calls > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Initial random evaluation to ensure best_x is set
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-5, 5, self.dim)
        f0 = counted_func(x0)
        
        # Phase 1: Differential Evolution
        de_budget = int(budget * 0.7)
        popsize = 15
        # Ensure we don't exceed budget
        maxiter = max(1, (de_budget - 1) // popsize) # -1 because we already did 1 call
        
        try:
            res_de = differential_evolution(
                counted_func, bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                seed=0,
                tol=1e-6
            )
        except Exception:
            pass

        if time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: Local Refinement
        remaining = budget - self.current_calls
        if remaining > 10:
            try:
                # Nelder-Mead
                res_local = minimize(
                    counted_func, best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
