"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, stagnation-based restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = None
        best_f = np.inf

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        calls_made = 1
        sigma = 4.0  # Large initial sigma to explore the box [-5, 5] quickly
        min_sigma = 1e-6
        max_sigma = 10.0
        
        stagnation_count = 0
        stagnation_limit_restart = 40  # Restart if stuck for this many steps
        early_budget_limit = 250       # Only allow restarts in the first 50% of budget

        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = x + step
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.5
                if sigma > max_sigma:
                    sigma = max_sigma
                stagnation_count = 0
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= 0.87
                if sigma < min_sigma:
                    sigma = min_sigma
                stagnation_count += 1
                
                # Restart logic:
                # If we are stuck (no improvement for stagnation_limit_restart steps)
                # and we are still early in the budget, restart from a new random point.
                if stagnation_count > stagnation_limit_restart and calls_made < early_budget_limit:
                    # Restart
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_made += 1
                    sigma = 4.0
                    stagnation_count = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Check time again after extra call
                    if time.time() > deadline:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
