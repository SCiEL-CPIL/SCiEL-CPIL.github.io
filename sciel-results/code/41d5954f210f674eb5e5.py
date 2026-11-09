"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 330s kill - 40s safety margin
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls_used = 1
        
        # Main (1+1)-ES loop with time guard
        while calls_used < self.budget:
            if time.time() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        
        return x, fx
# EVOLVE-BLOCK-END
