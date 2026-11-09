"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Initial point: center of domain
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 3.0  # Initial step size
        n_evals = 1
        
        for _ in range(self.budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            n_evals += 1
            
            if fy < fx:
                x, fx = y, fy
                if fy < best_fx:
                    best_x, best_fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            
            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
