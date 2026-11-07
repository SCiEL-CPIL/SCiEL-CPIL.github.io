import numpy as np
from scipy.optimize import basinhopping, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                return 1e100
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx
        
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        
        # Phase 1: Global Search with Basin Hopping
        # Use Nelder-Mead for local steps to avoid gradient overhead
        minimizer_kwargs = {'method': 'Nelder-Mead', 'maxiter': 10, 'options': {'xatol': 1e-3, 'fatol': 1e-3}}
        
        try:
            basinhopping(wrapped_func, x0, niter=15, minimizer_kwargs=minimizer_kwargs, 
                         stepsize=0.8, T=500.0)
        except Exception:
            pass
            
        # Phase 2: Local Polish
        if self.call_count < self.budget and self.best_x is not None:
            remaining = self.budget - self.call_count
            if remaining > 10:
                # Try L-BFGS-B for faster convergence on the smooth part
                try:
                    bounds = [(-5, 5)] * self.dim
                    minimize(wrapped_func, self.best_x, method='L-BFGS-B', 
                             options={'maxiter': remaining // 2, 'ftol': 1e-8, 'gtol': 1e-8},
                             bounds=bounds)
                except Exception:
                    pass
            else:
                # If very little budget left, do a few Nelder-Mead steps
                try:
                    minimize(wrapped_func, self.best_x, method='Nelder-Mead', 
                             options={'maxiter': remaining, 'xatol': 1e-5, 'fatol': 1e-5})
                except Exception:
                    pass
                    
        return self.best_x, self.best_f
