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
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Multi-start Nelder-Mead
        # Number of restarts
        n_restarts = 10
        # Evals per restart (approx)
        # 1 eval for initial point + NM evals
        evals_per_restart = max(1, (budget - n_restarts) // n_restarts)
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
                
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0
                
            # Run NM
            nm_count = [0]
            max_nm = evals_per_restart
            
            def nm_func(x):
                nm_count[0] += 1
                if nm_count[0] > max_nm:
                    return 1e10
                return func(x)
                
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_nm, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except:
                pass
                
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            
        return best_x, best_f
