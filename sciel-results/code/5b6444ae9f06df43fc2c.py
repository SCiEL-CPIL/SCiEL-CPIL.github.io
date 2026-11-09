import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Global Search with Latin Hypercube Sampling
        n_lhs = 100
        # Ensure we don't use all budget on LHS
        n_lhs = min(n_lhs, self.budget // 2)
        
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        points = sampler.random(n=n_lhs)
        # Map to [-5, 5]
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        
        # Evaluate LHS points
        for x in points:
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Remaining budget
        remaining_budget = self.budget - n_lhs
        
        # Define a wrapper to count calls? 
        # We can't easily stop scipy, but we can limit maxiter.
        # For dim=5, L-BFGS-B is very efficient.
        # Let's estimate: 1 iteration ~ 10-20 function calls (gradient + line search).
        # 400 calls / 15 ~ 26 iterations.
        
        bounds = [(-5, 5)] * self.dim
        
        try:
            res = minimize(
                func, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success and res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            # If scipy fails, keep the best from LHS
            pass
            
        return best_x, best_f
