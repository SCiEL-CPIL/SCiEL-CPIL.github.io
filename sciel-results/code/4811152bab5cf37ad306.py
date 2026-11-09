"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: stop 10 seconds before the 300s limit
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        
        # We have budget-1 iterations remaining
        for _ in range(self.budget - 1):
            # Check time before each expensive function call
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            # Check time after function call in case it took very long
            if time.time() > deadline:
                # We have consumed the time, but we have a valid fy.
                # We can update if it's better, but we must stop the loop.
                if fy < fx:
                    x, fx = y, fy
                break

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                
        return x, fx
# EVOLVE-BLOCK-END
