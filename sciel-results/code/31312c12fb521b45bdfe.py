"""Optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-restart and time guard for f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Multi-restart strategy: use 500 total evaluations
        # Split into restarts. For f6 (Attractive sector), the function has
        # a single global minimum but can have flat regions.
        # Strategy: first restart gets most budget, subsequent restarts get less.
        
        rng = np.random.default_rng(42)
        
        # Define restart schedule: how many evaluations per restart
        # First restart: 300 evals, second: 100, third: 100
        restart_budgets = [300, 100, 100]
        
        for restart_idx, r_budget in enumerate(restart_budgets):
            if best_f <= 1e-10:  # Good enough
                break
            
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Initialize: for first restart, use center; for others, random near best
            if restart_idx == 0:
                x = np.zeros(self.dim)
            else:
                # Restart near the best found so far with some perturbation
                x = best_x + 1.0 * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0 if restart_idx == 0 else 0.5
            evals_used = 1
            
            for i in range(r_budget - 1):
                # Time check every few iterations
                if i % 10 == 0 and time.time() > deadline - 5.0:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25  # Increase step on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8  # Decrease step on failure
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
