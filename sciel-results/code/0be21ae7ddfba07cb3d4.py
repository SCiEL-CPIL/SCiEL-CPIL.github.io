import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Start from a random point to handle hidden shifts
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        sigma = 2.0  # Initial step size
        
        for _ in range(budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            # Keep sigma in reasonable range
            if sigma > 5.0:
                sigma = 5.0
            elif sigma < 1e-4:
                sigma = 1e-4
        
        return best_x, best_fx
