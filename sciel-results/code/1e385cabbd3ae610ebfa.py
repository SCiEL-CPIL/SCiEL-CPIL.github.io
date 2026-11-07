import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, use 290s to be safe
        deadline = time.time() + 290.0
        
        # Fixed random seed for reproducibility and tuned performance
        rng = np.random.default_rng(42)
        
        # Initialize at center (0,0,...,0)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        stagnation_count = 0
        restart_threshold = 100  # Restart if no improvement in 100 steps
        
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x + sigma * rng.standard_normal(self.dim)
            # Clip to bounds to ensure valid input and keep state within box
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = func(y)
            calls_made += 1
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                
            # Adapt step size and check for stagnation
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
                stagnation_count = 0
            else:
                sigma *= 0.87
                stagnation_count += 1
                
            # Stagnation Restart: If stuck, jump to a new random point
            if stagnation_count >= restart_threshold:
                # Reset to a new random point in the likely minimizer region [-4, 4]^5
                x = rng.uniform(-4, 4, self.dim)
                fx = func(x)
                calls_made += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = 2.0
                stagnation_count = 0
                
                # Ensure we don't exceed budget
                if calls_made >= self.budget:
                    break

        return best_x, best_fx
