import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Random Exploration
        n_explore = 50
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            
        # Phase 2: Nelder-Mead Refinement
        # We have budget - calls_made left
        remaining_budget = budget - calls_made
        
        # Wrapper to count calls and enforce budget
        class CallCounter:
            def __init__(self, func, budget_limit):
                self.func = func
                self.count = 0
                self.budget_limit = budget_limit
                self.best_x = None
                self.best_f = np.inf
                
            def __call__(self, x):
                if self.count >= self.budget_limit:
                    return self.best_f
                self.count += 1
                fx = self.func(x)
                if fx < self.best_f:
                    self.best_f = fx
                    self.best_x = x
                return fx
                
        counter = CallCounter(func, remaining_budget)
        
        try:
            # Nelder-Mead
            # Initial simplex size: default is 0.05 * x0 + 0.00025 if x0 is 0
            # We should set initial_simplex based on our scale
            # Default options are usually fine, but maxiter is key.
            # Nelder-Mead does dim+1 evals per iter.
            # remaining_budget / (dim+1) is max iters.
            max_iter = max(1, remaining_budget // (dim + 1))
            
            res = minimize(counter, best_x, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                
        except Exception:
            pass
            
        return best_x, best_f
