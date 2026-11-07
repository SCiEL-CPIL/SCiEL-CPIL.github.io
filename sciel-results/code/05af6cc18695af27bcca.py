import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        # 1. Start at center
        x = np.zeros(dim)
        f_val = func(x)
        
        # 2. Adaptive coordinate-wise optimization
        # For each dimension, estimate gradient and move to optimal boundary
        h = 2.0  # Step size for gradient estimation
        for i in range(dim):
            if time.time() > deadline:
                break
            
            # Estimate gradient in dimension i from current position
            xp = x.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            
            # Determine direction based on function value change
            if fp > f_val:
                # Gradient is positive, move to lower bound
                x[i] = -5.0
            else:
                # Gradient is negative, move to upper bound
                x[i] = 5.0
            
            # Evaluate the improved point
            x = np.clip(x, -5, 5)
            f_val = func(x)
        
        # 3. Final verification (optional, ensures we're at the best point)
        if time.time() < deadline:
            f_val = func(x)
        
        return x, f_val
