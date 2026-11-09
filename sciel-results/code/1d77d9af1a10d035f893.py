import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        
        def check_time():
            return time.time() > deadline

        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1

        # 2. Coordinate descent: greedily fix each coordinate at its bound
        # For a linear function, the minimum over [-5,5]^5 is at a vertex.
        # We determine which bound minimizes f for each coordinate, given the others.
        # Start from center, and for each coordinate i, evaluate f at x_i=-5 and x_i=+5
        # (holding other coords at current values), then set x_i to the better bound.
        # This converges to a vertex in at most 2*dim = 10 calls.
        
        x_current = x_center.copy()
        f_current = f_center
        calls_used = 1

        for i in range(dim):
            if check_time() or calls_used >= budget:
                break
            
            # Evaluate at lower bound
            x_low = x_current.copy()
            x_low[i] = -5.0
            f_low = func(x_low)
            calls_used += 1
            
            if f_low < best_f:
                best_f = f_low
                best_x = x_low.copy()
            
            if check_time() or calls_used >= budget:
                break
            
            # Evaluate at upper bound
            x_high = x_current.copy()
            x_high[i] = 5.0
            f_high = func(x_high)
            calls_used += 1
            
            if f_high < best_f:
                best_f = f_high
                best_x = x_high.copy()
            
            # Fix coordinate i to the better bound
            if f_low <= f_high:
                x_current[i] = -5.0
            else:
                x_current[i] = 5.0

        # 3. Evaluate the final vertex
        if check_time() or calls_used >= budget:
            pass
        else:
            f_vertex = func(x_current)
            calls_used += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_current.copy()

        # 4. Fill remaining budget with the best point found
        while calls_used < budget:
            if check_time():
                break
            f_curr = func(best_x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr

        return best_x, best_f
