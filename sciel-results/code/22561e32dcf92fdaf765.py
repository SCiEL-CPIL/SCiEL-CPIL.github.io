import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial sampling to find a good start
        # Use 5 evaluations to pick the best of 5 random points
        n_init = min(5, self.budget)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                x_curr = x.copy()
                f_curr = fx
            
        if best_x is None:
             # Fallback if budget < 1, though unlikely
             x_curr = np.zeros(self.dim)
             f_curr = func(x_curr)
             best_x = x_curr.copy()
             best_f = f_curr
             
        calls = n_init
        
        sigma = 3.0 # Larger initial sigma
        sigma_min = 1e-6
        stagnation_limit = 25
        stagnation_counter = 0
        
        while calls < self.budget:
            # Time check
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x_curr + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < f_curr:
                # Success
                x_curr = y
                f_curr = fy
                sigma *= 1.5
                stagnation_counter = 0
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # Failure
                sigma *= 0.87
                stagnation_counter += 1
                
            # Check for restart condition
            if stagnation_counter >= stagnation_limit or sigma < sigma_min:
                # Restart by perturbing the best point found so far
                # Use a moderate perturbation
                x_curr = best_x + 1.0 * rng.standard_normal(self.dim)
                x_curr = np.clip(x_curr, -5, 5)
                f_curr = func(x_curr)
                calls += 1
                sigma = 2.0 # Reset sigma to a larger value to explore
                stagnation_counter = 0
                
                # If this new point is better, update best
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr.copy()
                    
                # If we ran out of budget in the restart eval, break
                if calls >= self.budget:
                    break

        return best_x, best_f
