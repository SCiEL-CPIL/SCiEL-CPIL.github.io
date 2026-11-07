"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Initial point
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Adaptive step size
        sigma = 1.5
        lambda_ = 10  # number of offspring per iteration
        
        # Parameters for step size adaptation
        c_sigma = 0.6  # adaptation speed
        target_success = 0.25  # target success rate for (1+lambda)-ES
        
        calls_made = 1
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            improvements = 0
            best_y = None
            best_fy = fx
            
            for j in range(lambda_):
                # Check time periodically
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                    
                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < best_fy:
                    best_fy = fy
                    best_y = y
                    improvements += 1
            
            # Adapt step size based on success rate
            actual_success = improvements / lambda_
            # Adjust sigma to approach target success rate
            # If too many successes, increase sigma; if too few, decrease
            if actual_success > target_success:
                sigma *= (1 + c_sigma * (actual_success - target_success))
            else:
                sigma *= (1 - c_sigma * (target_success - actual_success))
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Accept the best offspring
            if best_y is not None:
                x = best_y
                fx = best_fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
