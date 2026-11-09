import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 290.0 # 300s limit, keep 10s margin
        
        best_x = None
        best_f = np.inf
        
        # Number of restarts
        # Ensure total calls (1 init + N steps) * restarts <= budget
        n_restarts = 10
        # Reserve 2 calls per restart for safety margin (init + 1 extra check or just buffer)
        # (1 + steps) * 10 <= 500 => steps <= 49. Use 48 to be safe.
        budget_per_restart = (budget // n_restarts) - 2
        
        for r in range(n_restarts):
            # Initialize random point
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # SA parameters for this restart
            # Temperature should be scaled to the problem size
            T = 5.0 
            alpha = 0.95
            sigma = 1.5
            
            # Run SA for budget_per_restart steps
            for i in range(budget_per_restart):
                if time.time() - start_time > max_time:
                    break
                
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                delta = fy - fx
                
                # Acceptance probability
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adapt step size: increase on success
                    sigma *= 1.02
                else:
                    # Adapt step size: decrease on failure
                    sigma *= 0.98
                
                T *= alpha
            
            # Optional: If we have time, do a final polish on best_x?
            # The directive mentions "final high-precision polish pass".
            # We can use the remaining budget or just break.
            
        return best_x, best_f
