"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with diagonal covariance, adaptive step, and restart logic."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.5, self.dim)
        fx = func(x)
        
        # Diagonal covariance (independent step sizes per dimension)
        sigma = np.ones(self.dim) * 0.5
        
        # Adaptation parameters
        inc_factor = 1.15
        dec_factor = 0.85
        
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        best_x = x.copy()
        best_fx = fx
        no_improve_count = 0
        restart_threshold = 50
        
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
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Increase step sizes
                sigma *= inc_factor
            else:
                # Decrease step sizes
                sigma *= dec_factor
                no_improve_count += 1
                
                # If sigma becomes too small, we might be stuck.
                if np.min(sigma) < 1e-8:
                    sigma = np.ones(self.dim) * 0.3
            
            # Restart logic: if no improvement for a while, try a new random point
            if no_improve_count >= restart_threshold:
                # Try a few random restarts
                for _ in range(3):
                    x_new = rng.normal(0, 2.0, self.dim)
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = func(x_new)
                    if fx_new < best_fx:
                        x = x_new
                        fx = fx_new
                        best_x = x_new.copy()
                        best_fx = fx_new
                        no_improve_count = 0
                        sigma = np.ones(self.dim) * 0.5
                        break
                else:
                    # If no restart helped, keep exploring from current x with larger sigma
                    sigma = np.ones(self.dim) * 0.8
                    no_improve_count = 0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
