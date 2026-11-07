import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        bounds = [(-5, 5)] * dim
        
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]
        
        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise BudgetExceeded()
            
            call_count[0] += 1
            fx = func(x)
            
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        try:
            res = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=45, 
                popsize=10, 
                seed=42, 
                polish=False, 
                tol=0.01,
                init='latinhypercube',
                mutation=(0.5, 1.0)
            )
        except (BudgetExceeded, TimeoutError):
            pass
        except Exception:
            pass

        if best_x[0] is not None and call_count[0] < budget:
            try:
                res2 = minimize(
                    wrapped_func, 
                    best_x[0], 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5}
                )
            except Exception:
                pass

        return best_x[0], best_f[0]
