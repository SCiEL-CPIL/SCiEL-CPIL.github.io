import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = None
        best_f = np.inf
        
        # Phase 1: Diverse exploration (first ~50 calls)
        # Try several starting points with different strategies
        
        # Start with a point near origin (since shift is in [-4,4])
        x = rng.uniform(-3, 3, dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        calls = 1
        
        sigma = 1.5
        no_improve = 0
        
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2  # expand step
                no_improve = 0
            else:
                sigma *= 0.9  # shrink step
                no_improve += 1
            
            # Update best
            if fx < best_f:
                best_x, best_f = x.copy(), fx
            
            # If stuck for too long, do a random restart
            if no_improve > 30:
                # Random restart to a new point
                x = rng.uniform(-4, 4, dim)
                fx = func(x)
                calls += 1
                sigma = 1.5
                no_improve = 0
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
        
        return best_x, best_f
