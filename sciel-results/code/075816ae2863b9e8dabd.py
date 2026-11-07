# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Initial random sampling followed by adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s kill

        # 1. Initial Sampling to find a good starting region
        rng = np.random.default_rng(42)
        
        # Use 10 diverse points to locate a good basin
        n_init = 10
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        
        idx_best = np.argmin(F_init)
        x = X_init[idx_best].copy()
        fx = F_init[idx_best]
        
        calls_made = n_init

        # 2. Adaptive (1+1)-ES
        # Start with a moderate sigma. The (1+1)-ES will adapt it.
        # If the initial points were close, sigma can be small. 
        # If they were spread, sigma should be larger.
        # Let's start with 1.0 and let the success rule handle it.
        sigma = 1.0
        
        # Ensure we have budget left
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                # Success: increase step size
                # 1.5 is a standard aggressive factor for (1+1)
                sigma *= 1.5
            else:
                # Failure: decrease step size
                # 0.87 is a standard conservative factor
                sigma *= 0.87
            
            # Safety bounds for sigma
            if sigma < 1e-8:
                sigma = 1e-8
            if sigma > 10.0:
                sigma = 10.0

        return x, fx
# EVOLVE-BLOCK-END
