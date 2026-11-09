"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and adaptive stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within the instance
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        
        # Parameters for 1/5 success rule
        # Theoretical optimal for (1+1)-ES
        sigma_inc = np.sqrt(2.2)
        sigma_dec = 1.0 / np.sqrt(2.2)
        
        # Stagnation threshold: if no improvement for this many steps, restart
        stagnation_threshold = 15
        
        calls_made = 0
        max_sigma = 10.0
        min_sigma = 1e-5
        
        # Initial point: Start at origin as it's a common good start for BBOB
        # but for Rastrigin, origin is a local min (value A*(d-1)+A). 
        # Actually, for standard Rastrigin, origin is a local min. 
        # The global min is at 0.
        # So starting at 0 is actually starting at the global minimum!
        # Wait, the task says "hidden seeded shift, rotation and offset".
        # So the global minimum is NOT at 0. It is shifted.
        # So starting at 0 is just a random point in the shifted landscape.
        
        x = np.zeros(dim)
        fx = func(x)
        calls_made += 1
        
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
            
        sigma = 2.0
        stagnation_count = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Improvement
                x = y
                fx = fy
                sigma *= sigma_inc
                stagnation_count = 0
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement
                sigma *= sigma_dec
                stagnation_count += 1
                
            # Clamp sigma
            if sigma > max_sigma:
                sigma = max_sigma
            if sigma < min_sigma:
                sigma = min_sigma
                
            # Check for stagnation and restart
            if stagnation_count >= stagnation_threshold:
                # Restart logic
                # We are stuck in a local minimum.
                # Strategy: Jump to a new random point.
                # To maintain some continuity, we could jump near the best, 
                # but for Rastrigin, random jumps are better for escaping deep local mins.
                
                # If we have found a very good solution (relative to the start), 
                # we might want to try to polish it first.
                # But since we don't know f_opt, we just use the stagnation rule.
                
                # New random point
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Reset sigma to a moderate value for exploration
                sigma = 2.0
                stagnation_count = 0
                
                # If we run out of budget during the restart evaluation, break
                if calls_made >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
