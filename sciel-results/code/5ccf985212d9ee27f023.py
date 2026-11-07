import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s margin before 300s limit
        
        N = self.dim
        budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(N)
        
        bounds = [(-5.0, 5.0)] * N
        
        def eval_func(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                raise Exception("Budget exhausted")
            calls += 1
            return func(np.clip(x, -5.0, 5.0))
        
        def time_ok():
            return time.time() < deadline and calls < budget - 20
        
        # Phase 1: Multi-start SLSQP
        n_starts = 8
        for i in range(n_starts):
            if not time_ok():
                break
            
            x0 = np.random.uniform(-4.0, 4.0, N)
            
            try:
                res = minimize(eval_func, x0, method='SLSQP',
                               bounds=bounds,
                               options={'maxiter': 40, 'ftol': 1e-8, 'maxfun': 200})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        # Phase 2: Final high-precision polish from best point
        if time_ok() and best_f < np.inf:
            try:
                res = minimize(eval_func, best_x, method='SLSQP',
                               bounds=bounds,
                               options={'maxiter': 60, 'ftol': 1e-14, 'maxfun': 300})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        # Phase 3: One more polish with even tighter tolerance if budget remains
        if time_ok() and best_f < np.inf:
            try:
                res = minimize(eval_func, best_x, method='SLSQP',
                               bounds=bounds,
                               options={'maxiter': 30, 'ftol': 1e-16, 'maxfun': 150})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        return best_x
