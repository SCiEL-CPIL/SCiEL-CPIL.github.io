"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with diagonal covariance adaptation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        # Better initialization: Rosenbrock minimum is at (1,1,...,1), 
        # but hidden shift/rotation makes exact location unknown. 
        # Starting near 0 is generally safer than uniform random in [-5,5] 
        # for functions with a single central basin of attraction.
        x = rng.normal(0, 0.5, self.dim)
        fx = func(x)
        
        # Diagonal covariance (independent step sizes per dimension)
        sigma = np.ones(self.dim)
        
        # Adaptation parameters
        # Standard (1+1)-ES uses 1/5 rule, but for diagonal we can tune rates.
        # Increasing step size on success, decreasing on failure.
        inc_factor = 1.2
        dec_factor = 0.85
        
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase step sizes
                sigma *= inc_factor
            else:
                # Decrease step sizes
                sigma *= dec_factor
                
                # Optional: If sigma becomes too small, we might be stuck.
                # We can restart sigma to a moderate value to explore again.
                if np.min(sigma) < 1e-6:
                    sigma = np.ones(self.dim) * 0.5
                    # Optionally move x slightly to escape local minima
                    # But for Rosenbrock, just increasing sigma is usually enough 
                    # if we are in the valley. If we are stuck in a flat spot, 
                    # larger sigma helps.
        
        return x, fx
# EVOLVE-BLOCK-END
