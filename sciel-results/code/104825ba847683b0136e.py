import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        bounds = [(-5, 5)] * self.dim
        
        # Allocate 80% of budget to DE, 20% to local polish
        de_budget = int(self.budget * 0.8)
        polish_budget = self.budget - de_budget
        
        # DE parameters
        # popsize * maxiter <= de_budget
        # Let popsize = 10, maxiter = de_budget // 10
        popsize = 10
        maxiter = de_budget // popsize
        if maxiter < 1: maxiter = 1
        
        try:
            result = differential_evolution(
                func, 
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42,
                tol=1e-8,
                polish=False
            )
            best_x = result.x
            best_f = result.fun
        except Exception:
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            
        # Polish with Nelder-Mead
        if polish_budget > 10:
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': polish_budget * self.dim, 
                             'maxfev': polish_budget,
                             'xatol': 1e-6,
                             'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass
        
        return best_x, best_f
