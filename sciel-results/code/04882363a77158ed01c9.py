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
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Evaluate Center
        x = np.zeros(dim)
        f_curr = func(x)
        calls_made += 1
        update_best(x, f_curr)

        # 2. Greedy Coordinate Descent
        # For each axis, determine the optimal bound (-5 or 5) and move there immediately.
        # This updates the best-found value at each step, improving AOCC.
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Probe +5 on axis i
            x_plus = x.copy()
            x_plus[i] = 5.0
            f_plus = func(x_plus)
            calls_made += 1
            update_best(x_plus, f_plus)
            
            if f_plus < f_curr:
                x = x_plus
                f_curr = f_plus
            else:
                # Probe -5 on axis i
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                x_minus = x.copy()
                x_minus[i] = -5.0
                f_minus = func(x_minus)
                calls_made += 1
                update_best(x_minus, f_minus)
                
                if f_minus < f_curr:
                    x = x_minus
                    f_curr = f_minus
                # else: no improvement (should not happen for linear)

        # 3. Return best
        return best_x, best_f
