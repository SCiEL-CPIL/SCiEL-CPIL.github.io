import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        self.calls += 1
        
        x_best = x.copy()
        fx_best = fx
        
        sigma = 2.0
        no_improve = 0
        
        while self.calls < self.budget:
            # Time guard
            if time.time() - self.start_time > 250: # Safety margin
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            self.calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3 # Expand search
                no_improve = 0
                
                if fx < fx_best:
                    x_best = x.copy()
                    fx_best = fx
            else:
                sigma *= 0.85 # Shrink search
                no_improve += 1
                
            # Restart if stuck
            if no_improve > 40:
                # Restart from a new random point to explore new basins
                # But keep the best point found so far
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                self.calls += 1
                sigma = 2.0
                no_improve = 0
                
                if fx < fx_best:
                    x_best = x.copy()
                    fx_best = fx
        
        return x_best, fx_best
