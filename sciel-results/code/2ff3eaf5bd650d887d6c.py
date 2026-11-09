import numpy as np

class Optimizer:
    """(1+1)-ES with restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf
        remaining_budget = self.budget
        
        # Phase 1: Global Search with Restarted (1+1)-ES
        num_restarts = 3
        budget_per_restart = remaining_budget // num_restarts
        
        for _ in range(num_restarts):
            if remaining_budget <= 0:
                break
            
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            remaining_budget -= 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0
            local_budget = min(budget_per_restart, remaining_budget)
            
            for _ in range(local_budget):
                if remaining_budget <= 0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1 # Slightly more aggressive growth
                else:
                    sigma *= 0.9 # Slightly slower decay
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Phase 2: Local Polish
        if best_x is not None and remaining_budget > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1 # Start with small step
            
            for _ in range(remaining_budget):
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
