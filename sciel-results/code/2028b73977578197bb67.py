"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate step from standard normal
            z = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            
            if fy < fx:
                # Success
                x, fx = y, fy
                sigma *= 1.05  # Slight increase, conservative
            else:
                # Failure
                sigma *= 0.95  # Slight decrease
                
        return x, fx
# EVOLVE-BLOCK-END
