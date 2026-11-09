"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim
        
        # Use multiple restarts to escape local minima
        max_restarts = 15
        calls_per_restart = max(1, budget // max_restarts)
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize with random point, biased toward center for first restarts
            if restart < 3:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            # Adaptive sigma: start larger, decay faster
            sigma = 3.0 if restart == 0 else 2.0
            
            # (1+1)-ES with 1/5 success rule
            no_improve = 0
            for step in range(calls_per_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, 8.0)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.85, 0.005)
                    no_improve += 1
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Early restart if stuck for too long
                if no_improve > 20:
                    break
            
            # If we found a very good solution, we can stop early
            if best_f < -80:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
