import numpy as np
import time
from scipy.optimize import minimize,sobol
import scipy.stats as stats

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # Phase 1: Global search with Sobol sequence
        # 50 points is a good trade-off for 500 budget
        n_global = 50
        # scipy.stats.qmc.Sobol is available in newer scipy versions
        # If not, use rng
        try:
            sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n_global)
        except:
            points = rng.uniform(0, 1, (n_global, self.dim))
            
        # Map [0,1] to [-5,5]
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        
        calls = 0
        for p in points:
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p
        
        if best_x is None:
             # Fallback
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             calls += 1

        # Phase 2: Local refinement with Nelder-Mead
        remaining_budget = self.budget - calls
        
        # Wrap func to count and stop
        eval_count = [0]
        def wrapped_func(x):
            if time.time() > deadline:
                raise ValueError("Time limit")
            if eval_count[0] >= remaining_budget:
                raise ValueError("Budget limit")
            val = func(x)
            eval_count[0] += 1
            return val

        try:
            res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                           options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except (ValueError, Exception):
            pass # Keep best from global phase or partial local

        return best_x, best_f
