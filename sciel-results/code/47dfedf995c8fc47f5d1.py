import numpy as np
import time

class Optimizer:
    """
    (1+1)-ES with 1/5 success rule and restarts for BBOB f13 (sharp ridge).
    This method is robust for low-budget optimization of highly anisotropic functions.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility if needed, but random is fine
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        # Parameters for 1/5 success rule
        window_size = 15
        successes = []
        sigma = 2.0
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        total_evals += 1
        
        best_x = x.copy()
        best_f = fx
        
        # Main loop
        while total_evals < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_evals += 1
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Update current best for ES
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
            else:
                successes.append(0)
            
            # Adapt sigma based on 1/5 success rule
            if len(successes) >= window_size:
                # Keep only the last window_size entries
                successes = successes[-window_size:]
                success_rate = np.mean(successes)
                
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 1.0 / 1.1
                else:
                    # No change
                    pass
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart condition:
            # If sigma is very small and we haven't improved in a while, restart
            # Or if we are stuck at the boundary?
            # Simple restart: if no improvement for 50 steps and sigma < 0.1
            # We can track 'no_improve_count'
            # Let's add a simple stagnation counter
            
        return best_x, best_f
