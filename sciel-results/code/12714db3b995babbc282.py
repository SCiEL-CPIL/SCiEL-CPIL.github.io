"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Robust optimizer combining CMA-ES-like strategies with fallbacks."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Strategy 1: Try a well-tuned CMA-ES-like approach
        # Since we can't import cma directly, we implement a simplified version
        # or use differential_evolution from scipy as a robust baseline
        
        # Actually, let's use scipy's differential_evolution which is robust
        # but might be slow. Let's try a custom approach instead.
        
        # Custom (1+lambda)-ES with adaptive step size
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.0
        lambda_ = 10  # number of offspring
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate lambda offspring
            offspring = x + sigma * rng.standard_normal((lambda_, self.dim))
            offspring = np.clip(offspring, -5, 5)
            
            best_offspring_idx = 0
            best_offspring_fx = float('inf')
            
            for j in range(lambda_):
                f_val = func(offspring[j])
                if f_val < best_offspring_fx:
                    best_offspring_fx = f_val
                    best_offspring_idx = j
                
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = offspring[j].copy()
            
            if best_offspring_fx < fx:
                # Accept the best offspring
                x = offspring[best_offspring_idx].copy()
                fx = best_offspring_fx
                sigma *= 1.2  # Increase step size
            else:
                sigma *= 0.8  # Decrease step size
            
            sigma = max(sigma, 1e-4)  # Minimum step size
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
