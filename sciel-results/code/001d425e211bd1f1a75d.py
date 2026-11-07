import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        x0 = np.zeros(self.dim)
        fx0 = func(x0)
        best_x = x0
        best_fx = fx0
        
        # Phase 1: Nelder-Mead
        try:
            res = minimize(func, x0, method='Nelder-Mead', 
                           options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_fx:
                best_fx = res.fun
                best_x = res.x
        except Exception:
            pass
            
        # Phase 2: (1+1)-ES Polish
        # Estimate calls used? Hard.
        # Assume Nelder-Mead used ~300 calls.
        # We have ~200 left.
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5 # Small step for polish
        
        # Run (1+1)-ES for remaining budget
        # We don't know exact calls, so we run for a fixed number of steps
        # that is safe. 100 steps.
        for _ in range(100):
            if time.time() > deadline:
                break
                
            y = np.clip(x + sigma * np.random.randn(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            if fx < best_fx:
                best_fx = fx
                best_x = x
                
        return best_x, best_fx
