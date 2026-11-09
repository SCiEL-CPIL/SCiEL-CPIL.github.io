"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        # Initial broad sampling to find a good starting point
        n_initial = 30
        best_x = None
        best_f = np.inf

        for _ in range(n_initial):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                # Check time
                if time.time() > deadline:
                    return best_x, best_f

        # Multi-start (1+1)-ES from the best initial point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        calls_used = n_initial
        remaining = self.budget - calls_used

        # Adaptively choose number of restarts based on remaining budget
        # We'll do a series of short (1+1)-ES runs with restarts
        # Each restart gets a fixed portion of the remaining budget
        
        # Strategy: do several short ES runs, restarting if no improvement
        # for a while, with increasing perturbation of the best point
        
        restarts = 0
        max_restarts = 5
        
        for restart in range(max_restarts):
            # For each restart, do a (1+1)-ES with limited budget
            # Budget per restart: distribute remaining calls
            if remaining <= 0:
                break
            
            # Use a portion of remaining budget for this restart
            # Early restarts get more budget
            frac = 0.6 if restart == 0 else 0.2
            steps_this = int(remaining * frac)
            if steps_this < 5:
                steps_this = max(5, remaining)
            if steps_this > remaining:
                steps_this = remaining
            
            # If this is not the first restart, perturb the best point
            if restart > 0:
                # Perturb with decreasing magnitude
                perturb_sigma = 2.0 / (restart + 1)
                x = np.clip(best_x + perturb_sigma * rng.standard_normal(self.dim), -5, 5)
                fx = func(x)
                calls_used += 1
                remaining -= 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Run (1+1)-ES for steps_this iterations
            local_best_x = x.copy()
            local_best_f = fx
            local_sigma = 1.0 if restart == 0 else 0.5
            
            for step in range(steps_this):
                if time.time() > deadline:
                    return best_x, best_f
                
                y = np.clip(local_best_x + local_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                remaining -= 1
                
                if fy < local_best_f:
                    local_best_x = y
                    local_best_f = fy
                    local_sigma *= 1.3
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                else:
                    local_sigma *= 0.85
                
                if remaining <= 0:
                    break
            
            if remaining <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
