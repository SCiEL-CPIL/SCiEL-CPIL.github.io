import numpy as np
import time
import scipy.optimize
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin for 300s limit
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.bounds = [(-5.0, 5.0)] * self.dim

        def wrapped_func(x):
            # Check time
            if time.time() > deadline:
                return 1e100 # Return a very large value to stop optimization
            
            # Check budget
            if self.call_count >= self.budget:
                return 1e100
            
            self.call_count += 1
            
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            
            # Clip to box just in case, although func does it too
            x = np.clip(x, -5.0, 5.0)
            
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e100
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # Phase 1: Differential Evolution
        # popsize=12, maxiter=32 -> ~12 + 32*12 = 400 calls
        de_popsize = 12
        de_maxiter = 32
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                self.bounds, 
                popsize=de_popsize, 
                maxiter=de_maxiter, 
                polish=False, 
                init='sobol', 
                seed=0,
                tol=1e-12
            )
        except Exception:
            res_de = None

        # If DE failed or didn't run, start from center
        if res_de is None or not hasattr(res_de, 'x'):
            x0 = np.zeros(self.dim)
        else:
            x0 = res_de.x

        # Phase 2: Local Polish with Nelder-Mead
        # Remaining budget
        remaining_budget = self.budget - self.call_count
        
        if remaining_budget > 10 and time.time() < deadline:
            # Define a wrapper that stops when budget is hit
            polish_count = [0]
            
            def polish_func(x):
                if polish_count[0] >= remaining_budget:
                    return 1e100
                polish_count[0] += 1
                return wrapped_func(x)

            try:
                res_nm = minimize(
                    polish_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_nm.fun < self.best_f:
                    self.best_x = res_nm.x
                    self.best_f = res_nm.fun

            except Exception:
                pass

        # Final return
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)
            
        return self.best_x, self.best_f
