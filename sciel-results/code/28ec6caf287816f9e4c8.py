import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        n_restarts = 4
        budget_per_restart = budget // n_restarts
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
                
            x0 = rng.uniform(-5, 5, D)
            
            # Create initial simplex
            # x0, x0 + delta*e_i
            delta = 1.0
            simplex = [x0]
            for j in range(D):
                x_j = x0.copy()
                x_j[j] += delta
                simplex.append(x_j)
                
            calls_used = 0
            max_calls = budget_per_restart
            
            def wrapped_func(x):
                nonlocal calls_used
                calls_used += 1
                if calls_used > max_calls:
                    return np.inf
                return func(x)
                
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'initial_simplex': np.array(simplex), 
                                        'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
        if best_x is None:
            best_x = np.zeros(D)
            best_f = func(best_x)
            
        return best_x, best_f
