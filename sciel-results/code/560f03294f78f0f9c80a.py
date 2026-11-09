"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, multiple restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Use multiple restarts to explore the landscape
        # For BBOB f6 (Attractive sector), the landscape has multiple local optima
        # We'll do several short runs with different starting points
        
        # Budget allocation: spend most of the budget on the best-performing strategy
        # but also explore with restarts
        
        # Strategy: Run (1+1)-ES with adaptive sigma, restart if stuck
        # Track progress to detect stagnation
        
        total_budget = self.budget
        # Reserve some budget for potential restarts
        # We'll do up to 3 restarts if we detect stagnation
        
        max_restarts = 3
        restart_threshold = 80  # If no improvement in this many steps, consider restart
        
        for restart in range(max_restarts + 1):
            if time.time() > deadline:
                break
            
            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Start near the best point found so far with some perturbation
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(self.dim) * 1.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            calls_in_run = 1
            no_improve_count = 0
            run_best_f = fx
            
            # Adaptive step size (1+1)-ES
            while calls_in_run < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_in_run += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05  # Slightly increase step size on success
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.97  # Slightly decrease step size on failure
                    no_improve_count += 1
                
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-6, 8.0)
                
                # Check for stagnation
                if no_improve_count >= restart_threshold and restart < max_restarts:
                    # Try a restart
                    if time.time() > deadline:
                        break
                    break
                
                # If sigma is very small, we're likely converged
                if sigma < 1e-4:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
