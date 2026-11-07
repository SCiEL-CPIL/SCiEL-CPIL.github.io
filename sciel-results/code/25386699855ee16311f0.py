"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # Multi-start strategy to handle multimodality of Rastrigin
        n_starts = 5
        budget_per_start = self.budget // n_starts
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_starts):
            # Check time
            if time.time() > deadline:
                break
            
            # Use different seeds for diversity
            rng = np.random.default_rng(i + 1)
            
            # Initial point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if budget_per_start <= 1:
                continue
                
            sigma = 2.0
            remaining = budget_per_start - 1
            
            for _ in range(remaining):
                # Check time inside the loop as well
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
        
        # Fallback if no calls were made (should not happen)
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
