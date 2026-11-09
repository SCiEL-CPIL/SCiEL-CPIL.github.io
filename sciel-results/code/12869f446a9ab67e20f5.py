import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        x = np.zeros(dim)
        best_f = func(x)
        calls_used = 1
        
        # Sequential coordinate-wise vertex construction
        for i in range(dim):
            if time.time() > deadline:
                break
                
            # Try +5
            x_plus = x.copy()
            x_plus[i] = 5.0
            f_plus = func(x_plus)
            calls_used += 1
            
            if f_plus < best_f:
                best_f = f_plus
                x = x_plus
            else:
                # Try -5
                x_minus = x.copy()
                x_minus[i] = -5.0
                f_minus = func(x_minus)
                calls_used += 1
                
                # For linear function, one of +5 or -5 must be better than 0
                # So f_minus should be < best_f (which was f(0) or improved)
                # Update x and best_f
                x = x_minus
                best_f = f_minus
                
        return x, best_f
