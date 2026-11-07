"""Rewritten program for the suite task; uses Random Restart Hill Climbing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Random Restart Hill Climber with Adaptive Step Sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        # We will do restarts until budget or time runs out.
        # Each restart starts with a random point and a large step size.
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize a new trajectory
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # If we ran out of budget, stop
            if calls_made >= self.budget:
                break
                
            # Adaptive step size for this trajectory
            sigma = 2.0  # Start with a large step to jump over plateaus
            
            # Local search loop
            while calls_made < self.budget:
                if time.time() > deadline:
                    break
                    
                # Generate a candidate
                # Use a Gaussian step, scaled by sigma
                direction = rng.standard_normal(self.dim)
                # Normalize direction to unit length to make sigma a pure distance scale
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    continue
                    
                y = x + sigma * direction
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Improvement: move to y
                    x = y
                    fx = fy
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Increase step size slightly to continue moving in the "good" direction
                    # But don't make it too huge
                    sigma = min(sigma * 1.2, 4.0) 
                else:
                    # No improvement: shrink step size significantly
                    # This allows fine-grained search if we are close to a local min
                    # or to try different directions if stuck
                    sigma *= 0.5
                    
                    # If sigma becomes too small, it's likely stuck in a flat region or local min
                    # We can break out of this local search and restart
                    if sigma < 0.01:
                        break
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
