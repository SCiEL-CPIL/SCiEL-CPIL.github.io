import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls = 1
        start_time = time.time()
        
        # Parameters for restart
        restart_threshold = 15
        restart_counter = 0
        
        while calls < self.budget:
            # Time guard
            if time.time() - start_time > 290:
                break
            
            # Generate neighbor
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
                restart_counter = 0
            else:
                sigma *= 0.87
                restart_counter += 1
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Restart check
            if restart_counter >= restart_threshold:
                # Only restart if we have budget left for the restart evaluation
                if calls < self.budget:
                    x = rng.uniform(-5, 5, self.dim)
                    fx = func(x)
                    calls += 1
                    sigma = 2.0
                    restart_counter = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    break

        return best_x, best_f
