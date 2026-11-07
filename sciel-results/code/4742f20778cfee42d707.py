import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        rng = np.random.default_rng(0)
        
        # Initial point: Random within the box [-5, 5]
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        # Initial step size
        sigma = 3.0
        min_sigma = 1e-10
        max_sigma = 10.0
        
        calls_made = 1
        best_x = x.copy()
        best_fx = fx
        
        # Stagnation tracking
        stagnation = 0
        max_stagnation = 40  # Restart if no improvement for this many iterations
        
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                stagnation = 0
                # More aggressive growth when improving
                sigma *= 1.3
                if sigma > max_sigma:
                    sigma = max_sigma
            else:
                sigma *= 0.9
                if sigma < min_sigma:
                    sigma = min_sigma
                stagnation += 1
            
            # Track best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            # Stagnation-based restart: jump to a new random point
            if stagnation >= max_stagnation:
                # Try a few random restarts
                for _restart in range(3):
                    if time.time() > deadline:
                        break
                    xr = rng.uniform(-5, 5, dim)
                    fxr = func(xr)
                    calls_made += 1
                    if fxr < best_fx:
                        best_fx = fxr
                        best_x = xr.copy()
                        x = xr
                        fx = fxr
                        sigma = 5.0  # Larger sigma to explore from new point
                        stagnation = 0
                        break
                else:
                    # No improvement from restarts, keep going with current
                    x = best_x
                    fx = best_fx
                    sigma = 2.0
                    stagnation = 0
        
        return best_x, best_fx
