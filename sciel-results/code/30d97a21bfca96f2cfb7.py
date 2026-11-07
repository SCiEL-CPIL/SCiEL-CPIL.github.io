import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # Start near origin since minimizer is in [-4,4]^5
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        # Diagonal covariance adaptation
        sigma = np.ones(dim)
        
        # Track recent step successes for adaptation
        success_count = 0
        window_size = 10
        
        for i in range(1, budget):
            if time.time() > deadline:
                break
            
            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x_best + step, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx_best:
                # Success
                x_best = y
                fx_best = fy
                success_count += 1
                
                # Increase sigma for dimensions that improved
                improved_mask = step != 0
                sigma[improved_mask] *= 1.15
            else:
                # Failure
                # Decrease sigma for all dimensions
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 5.0)
            
            # Periodic adaptation: if too many successes, increase; if too few, decrease
            if i % window_size == 0:
                if success_count / window_size > 0.5:
                    sigma *= 1.1
                elif success_count / window_size < 0.2:
                    sigma *= 0.8
                success_count = 0
        
        return x_best, fx_best
