"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Start with a larger sigma to quickly traverse the multimodal landscape
        # of Schwefel's function. The 1/5 rule will adapt it down.
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 3.0
        min_sigma = 1e-8
        max_sigma = 10.0
        
        calls_made = 1
        
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
                
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if sigma > max_sigma:
                    sigma = max_sigma
            else:
                sigma *= 0.87
                if sigma < min_sigma:
                    sigma = min_sigma
                    
        return x, fx
# EVOLVE-BLOCK-END
