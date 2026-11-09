import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_used = 1
        
        # Step 2: Sequential coordinate descent to find the optimal vertex
        # For a linear function f(x) = c + v^T x, the minimum is at a vertex
        # of the box [-5, 5]^dim. We can find which vertex by checking each
        # coordinate independently.
        
        # Current center (start at origin)
        center = np.zeros(dim)
        current_f = f0
        
        # For each dimension, determine the optimal bound
        for i in range(dim):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Evaluate at -5 and +5 for dimension i, keeping other dims at center
            x_neg = center.copy()
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            calls_used += 1
            
            if calls_used >= budget or time.time() > deadline:
                # Can't evaluate the positive side, assume negative is better
                center[i] = -5.0
                if f_neg < best_f:
                    best_f = f_neg
                    best_x = x_neg.copy()
                continue
            
            x_pos = center.copy()
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            calls_used += 1
            
            # Choose the better bound for this dimension
            if f_neg <= f_pos:
                center[i] = -5.0
                if f_neg < best_f:
                    best_f = f_neg
                    best_x = x_neg.copy()
            else:
                center[i] = 5.0
                if f_pos < best_f:
                    best_f = f_pos
                    best_x = x_pos.copy()
        
        # After fixing all coordinates, center is the optimal vertex
        # Evaluate it to confirm (it should already be best_f if we tracked correctly)
        if calls_used < budget and time.time() < deadline:
            f_vertex = func(center)
            calls_used += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = center.copy()
        
        # Use remaining budget to re-evaluate the best point
        # (this doesn't change best_f but keeps the call pattern stable)
        while calls_used < budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_used += 1
        
        return best_x, best_f
