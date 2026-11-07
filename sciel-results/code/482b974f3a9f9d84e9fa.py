import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start Powell
        # Use 5 restarts
        n_restarts = 5
        evals_per_restart = budget // n_restarts # 100
        
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Run Powell
            # Powell uses coordinate lines, so it's robust.
            res = minimize(counted_func, x0, method='Powell', options={'maxfev': evals_per_restart, 'xtol': 1e-8, 'ftol': 1e-8})
            
            # res.fun is the best value found by this run
            # Update best if better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        # Phase 2: Polish the best point with Nelder-Mead (faster local convergence)
        if calls_made < budget - 20 and time.time() < deadline - 5:
            # Use remaining budget for polishing
            remaining = budget - calls_made
            res = minimize(counted_func, best_x, method='Nelder-Mead', options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        return best_x, best_f
