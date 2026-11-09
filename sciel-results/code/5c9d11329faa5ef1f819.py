import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 325.0
        dim = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        used = 0
        
        def check_deadline():
            return time.time() > deadline

        rng = np.random.default_rng(12345)
        
        # Phase 1: Multi-start Nelder-Mead
        n_starts = 5
        calls_per_start = 80
        
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        for s in starts:
            if used >= total_budget or check_deadline():
                break
                
            x0 = np.clip(s, -5, 5)
            f0 = func(x0)
            used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            local_used = 0
            max_local_calls = calls_per_start
            local_best_x = x0.copy()
            local_best_f = f0
            
            def local_func(x):
                nonlocal local_used, local_best_x, local_best_f
                if local_used >= max_local_calls:
                    raise ValueError("Budget exceeded")
                local_used += 1
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                if val < local_best_f:
                    local_best_f = val
                    local_best_x = x_clipped.copy()
                return val
            
            try:
                res = minimize(local_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_local_calls, 'xatol': 1e-6, 'fatol': 1e-6})
            except ValueError:
                pass
            except Exception:
                pass
            
            used += local_used
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Phase 2: Polish with (1+1)-ES if budget remains
        remaining = total_budget - used
        if remaining > 10 and not check_deadline():
             x = best_x.copy()
             fx = best_f
             sigma = 0.5
             
             while used < total_budget and not check_deadline():
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = func(y)
                 used += 1
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.05
                     if fx < best_f:
                         best_f = fx
                         best_x = x.copy()
                 else:
                     sigma *= 0.9
                     if sigma < 1e-6:
                         break

        return best_x, best_f
