import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        
        # 1. Deterministic Sobol sampling for initial exploration
        # Use a fixed seed for reproducibility if needed, but scramble=True is fine
        n_sobol = min(50, budget // 2)
        
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = -5 + 10 * sobol_points
        
        best_x = sobol_points[0].copy()
        best_f = func(best_x)
        count = 1
        
        for i in range(1, n_sobol):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Refinement using scipy.optimize.minimize
        remaining_budget = budget - count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_f
            
        try:
            # Powell is a good derivative-free method for this dimension
            # maxfev ensures we don't exceed budget
            res = minimize(func, best_x, method='Powell', 
                           options={'maxfev': remaining_budget, 'maxiter': 1000})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
        
        return best_f
