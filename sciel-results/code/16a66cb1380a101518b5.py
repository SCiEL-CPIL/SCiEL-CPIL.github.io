import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration (Multi-Start)
        # Sample enough points to likely hit the vicinity of the ridge
        # 50 points is a good balance for 5D with 500 budget
        n_explore = 50
        points = rng.uniform(-5, 5, (n_explore, n))
        
        values = np.full(n_explore, np.inf)
        calls_made = 0
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            values[i] = func(points[i])
            calls_made += 1
            
        if calls_made == 0:
             # Fallback
             x0 = np.zeros(n)
             f0 = func(x0)
             return x0, f0

        # Find the best starting point
        best_idx = np.argmin(values)
        x_best = points[best_idx].copy()
        f_best = values[best_idx]
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_best, f_best

        # Phase 2: (1+1)-ES Refinement
        # Start from the best global point
        x_curr = x_best.copy()
        f_curr = f_best
        
        # Initial step size
        # The ridge is sharp, so we shouldn't start too large, but 1.0 is reasonable
        # for a box of size 10.
        sigma = 1.0
        
        # We will run (1+1)-ES for the remaining budget
        # Each step costs 1 evaluation
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Standard normal perturbation
            z = rng.standard_normal(n)
            x_new = x_curr + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < f_curr:
                # Success
                x_curr = x_new
                f_curr = f_new
                # Increase step size slightly
                # 1.1 is a gentle increase, allowing exploration
                sigma *= 1.1 
            else:
                # Failure
                # Decrease step size
                # 0.9 is a gentle decrease, allowing convergence
                sigma *= 0.9
                
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x_curr, f_curr
