"""(1+1)-Evolution Strategy with Adaptive Step Size and Diverse Restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a (1+1)-Evolution Strategy with adaptive step size.
    Includes a restart mechanism to escape local minima by jumping to new random points
    if stagnation is detected.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin from 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initialize
        x = rng.uniform(-5, 5, dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        
        # Parameters
        sigma = 1.0
        stagnation = 0
        max_stagnation_for_restart = 30
        restart_count = 0
        max_restarts = 5 # Limit restarts to avoid wasting budget on random points
        
        # Success rate tracking for step size adaptation
        # Target success rate ~ 1/sqrt(dim)
        target_success = 1.0 / np.sqrt(dim)
        # Step size adaptation factor
        # If success rate is high, increase sigma; if low, decrease.
        # We use a simple rule: 
        # Keep a short window of successes? Or just use the last step?
        # Standard (1+1)-ES uses a longer time constant.
        # Here, we use a simple exponential moving average or just the last 10 steps.
        
        # Let's use a simpler heuristic:
        # If last step was success, sigma *= 1.1
        # If last step was failure, sigma *= 0.9
        # This is aggressive but works well for low dims.
        
        evals_used = 1
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            f_y = func(y)
            evals_used += 1
            
            if f_y < f_x:
                # Success
                x = y
                f_x = f_y
                stagnation = 0
                
                # Update global best
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                
                # Increase sigma to explore more
                sigma = min(2.0, sigma * 1.05)
            else:
                # Failure
                stagnation += 1
                
                # Decrease sigma to exploit
                sigma = max(0.01, sigma * 0.95)
                
                # Check for restart
                if stagnation >= max_stagnation_for_restart:
                    restart_count += 1
                    if restart_count <= max_restarts:
                        # Restart from a new random point
                        # This consumes the current iteration's "logic" but we already used a call for y.
                        # We need to make a new random point.
                        # To save calls, we can just reset x and sigma, but x is currently the best of the local search.
                        # If we reset x to a random point, we lose the local best.
                        # Strategy: Only restart if the current best is not improving.
                        # We will pick a new random point as the new x.
                        x = rng.uniform(-5, 5, dim)
                        f_x = func(x) # This uses a call!
                        evals_used += 1
                        
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        
                        sigma = 1.0
                        stagnation = 0
                    else:
                        # If we've used all restarts, just continue with local search
                        # but maybe increase sigma to try to escape?
                        sigma = 1.5
                        stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
