import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Parameters
        num_restarts = 10
        budget_per_restart = self.budget // num_restarts
        rng = np.random.default_rng(0)
        
        # Safety margin: 280 seconds
        max_time = 280.0
        
        for r in range(num_restarts):
            # Check time
            if time.time() - self.start_time > max_time:
                break
            if self.eval_count >= self.budget:
                break
            
            # Initialize new random point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            self.eval_count += 1
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            sigma = 1.0
            
            # Run (1+1)-ES
            for step in range(budget_per_restart - 1):
                if time.time() - self.start_time > max_time:
                    break
                if self.eval_count >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                self.eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                else:
                    sigma *= 0.8
                    
        return self.best_x, self.best_f
