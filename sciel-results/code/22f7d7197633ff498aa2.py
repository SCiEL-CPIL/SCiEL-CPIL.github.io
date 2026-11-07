import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s margin before 330s kill
        rng = np.random.default_rng(0)
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # Best ever
        x_ever = np.zeros(dim)
        f_ever = np.inf
        
        # (mu+lambda)-ES parameters
        mu = 4
        lam = 16
        sigma = 2.0
        stall_counter = 0
        stall_threshold = 15
        
        # Initialize population
        pop = rng.uniform(-5, 5, (mu, dim))
        f_pop = np.array([func(x) for x in pop])
        calls += mu
        
        # Find best
        best_idx = np.argmin(f_pop)
        x_best = pop[best_idx].copy()
        f_best = f_pop[best_idx]
        x_ever = x_best.copy()
        f_ever = f_best
        
        while calls + lam <= budget and time.time() < deadline:
            # Generate offspring
            offspring = x_best + sigma * rng.standard_normal((lam, dim))
            offspring = np.clip(offspring, -5, 5)
            
            f_vals = np.array([func(x) for x in offspring])
            calls += lam
            
            best_idx = np.argmin(f_vals)
            best_val = f_vals[best_idx]
            
            if best_val < f_best:
                x_best = offspring[best_idx]
                f_best = best_val
                stall_counter = 0
                # Increase sigma on success
                sigma *= 1.1
            else:
                stall_counter += 1
                # Decrease sigma on failure
                sigma *= 0.9
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            if f_best < f_ever:
                f_ever = f_best
                x_ever = x_best.copy()
            
            # Restart if stalled
            if stall_counter > stall_threshold and calls < budget - 1 and time.time() < deadline:
                # Restart
                x_best = rng.uniform(-5, 5, dim)
                f_best = func(x_best)
                calls += 1
                stall_counter = 0
                sigma = 2.0
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
                
        # Use remaining budget if any
        while calls < budget and time.time() < deadline:
            # Simple (1+1) step
            y = x_best + sigma * rng.standard_normal(dim)
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
                sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 5.0)
                
        return x_ever, f_ever
