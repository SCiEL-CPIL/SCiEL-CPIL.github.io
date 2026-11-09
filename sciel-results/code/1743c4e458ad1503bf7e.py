"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restarting Simulated Annealing optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 290.0
        
        best_x = None
        best_f = np.inf
        
        # Configuration
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            # Initialize random point
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # SA parameters
            T = 10.0
            alpha = 0.99
            sigma = 2.0
            
            # Run SA
            for i in range(budget_per_restart):
                if time.time() - start_time > max_time:
                    return best_x, best_f
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                delta = fy - fx
                
                # Acceptance
                if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.01
                else:
                    sigma *= 0.99
                
                T *= alpha
            
            # If time remains, we could do a final polish, but SA is stochastic.
            # The restarts handle the global search.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
