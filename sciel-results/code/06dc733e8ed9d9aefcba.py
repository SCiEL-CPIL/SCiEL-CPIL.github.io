"""Optimized program for BBOB-style f6 (Attractive sector) using Multi-Start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 280 seconds is well within the 300s limit
        deadline = start_time + 280.0
        
        # Split budget among multiple restarts to explore different basins
        # 5 restarts of 100 evaluations each
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = float('inf')
        
        for r in range(n_restarts):
            # Check time before starting a new restart
            if time.time() > deadline:
                break
                
            # Use a different seed for each restart to ensure diversity
            # Base seed + restart index
            rng = np.random.default_rng(42 + r)
            
            # Initialize in the center or random? Random is better for diversity.
            # Clip to [-5, 5]
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Initial step size: large enough to move across the space
            # Dim is 5, box size is 10. sigma=2 is a good start.
            sigma = 2.0
            
            # (1+1)-ES loop for this restart
            for i in range(budget_per_restart):
                # Time check inside the loop
                if time.time() > deadline:
                    break
                    
                # Generate offspring
                # Standard normal distribution
                step = sigma * rng.standard_normal(self.dim)
                y = x + step
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                
                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                # Selection and Step Size Adaptation (1/5 success rule)
                # If improvement, increase sigma; else decrease.
                # Factors: 1.5 for success, 0.85 for failure (standard approximations)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                else:
                    sigma *= 0.85
                
                # Prevent sigma from becoming too small or too large
                if sigma < 1e-8:
                    sigma = 1e-8
                if sigma > 10.0:
                    sigma = 10.0

        # If best_x is None (should not happen if at least 1 call made), return a default
        if best_x is None:
             # Fallback: single call
             rng = np.random.default_rng(0)
             x = rng.uniform(-5, 5, self.dim)
             best_x = x
             best_f = func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
