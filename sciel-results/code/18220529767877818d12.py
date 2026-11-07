import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # 30s margin before 320s limit
        
        # Initial point
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # Best ever
        x_ever = x_best.copy()
        f_ever = f_best
        
        sigma = 2.0
        lambda_pop = 5
        stall_counter = 0
        stall_threshold = 15  # 15 generations * 5 offspring = 75 calls without improvement
        
        while calls + lambda_pop <= self.budget:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring = x_best + sigma * rng.standard_normal((lambda_pop, self.dim))
            offspring = np.clip(offspring, -5, 5)
            
            f_vals = np.array([func(x) for x in offspring])
            calls += lambda_pop
            
            best_idx = np.argmin(f_vals)
            best_val = f_vals[best_idx]
            
            if best_val < f_best:
                x_best = offspring[best_idx]
                f_best = best_val
                stall_counter = 0
                # Increase sigma on success
                sigma *= 1.15
            else:
                stall_counter += 1
                # Decrease sigma on failure
                sigma *= 0.85
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            if f_best < f_ever:
                f_ever = f_best
                x_ever = x_best.copy()
            
            # Restart if stalled
            if stall_counter > stall_threshold and calls < self.budget - 1:
                # Restart
                x_best = rng.uniform(-5, 5, self.dim)
                f_best = func(x_best)
                calls += 1
                stall_counter = 0
                sigma = 2.0
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
                
        # Use remaining budget if any
        while calls < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Simple (1+1) step
            y = x_best + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            if fy < f_best:
                x_best = y
                f_best = fy
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
            else:
                sigma *= 0.85
                sigma = np.clip(sigma, 1e-6, 5.0)
                
        return x_ever, f_ever
