import numpy as np
import time
import scipy.optimize
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 10 seconds safety margin
        deadline = start_time + 290.0
        
        # Determine number of initial points
        # Use a fixed reasonable number, capped by budget
        n_init = 64
        if self.budget < n_init + 50:
            n_init = max(1, self.budget // 2)
            
        # Generate Sobol points
        # Sobol requires d < 21201 or so, dim=5 is fine.
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True)
        # We need to generate enough points.
        # Sobol generator produces points in [0, 1)^d.
        points = sampler.random(n_init)
        
        # Map to [-5, 5]^d
        x_init = points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Evaluate initial points
        for x in x_init:
            if time.time() > deadline:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            if calls >= self.budget:
                break

        # If we didn't make any calls (should not happen), return a default
        if best_x is None:
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             calls += 1

        # Local optimization with Powell
        if calls < self.budget and time.time() < deadline:
            remaining_budget = self.budget - calls
            # We subtract 1 to be safe, in case powell counts differently or we need a buffer
            # But maxfev is strict.
            try:
                res = scipy.optimize.powell(
                    func, 
                    best_x, 
                    maxfev=remaining_budget
                )
                # Update best if powell found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If powell fails, keep the best from initialization
                pass

        return best_x, best_f
