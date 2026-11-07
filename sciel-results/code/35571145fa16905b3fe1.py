import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initialize with a single random point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        sigma = 2.0
        budget_left = self.budget - 1
        
        # (1+1)-ES with adaptive step size
        for _ in range(budget_left):
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                # Improvement: move to new point, increase step size
                x, fx = y, fy
                sigma *= 1.5
            else:
                # No improvement: decrease step size
                sigma *= 0.87
                
        return x, fx
