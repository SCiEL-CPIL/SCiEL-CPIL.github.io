"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill
        
        n = self.dim
        total_budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial point: origin is a common prior for BBOB
        x = np.zeros(n)
        fx = safe_func(x)
        best_x = x.copy()
        best_f = fx
        
        sigma = 2.0
        no_improve = 0
        max_no_improve = 50
        
        # Adaptive step size logic
        while calls_made < total_budget and (time.time() - start_time) < deadline:
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = safe_func(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase step size on success
                sigma *= 1.3
                no_improve = 0
            else:
                # Decrease step size on failure
                sigma *= 0.7
                no_improve += 1
                
            # Restart if stuck
            if no_improve > max_no_improve:
                # Reset to best found so far
                x = best_x.copy()
                fx = best_f
                sigma = 2.0
                no_improve = 0
                max_no_improve = 50  # Reset threshold
                # Optional: slight random perturbation around best
                # x = best_x + 0.5 * rng.standard_normal(n)
                # x = np.clip(x, -5, 5)
                # fx = safe_func(x)
                # break # Let it continue from best
                
        return best_x, best_f
# EVOLVE-BLOCK-END
