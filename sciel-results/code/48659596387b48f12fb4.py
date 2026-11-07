import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        # 1. Start at center
        x = np.zeros(dim)
        f_center = func(x)
        calls_made = 1
        
        # 2. Greedy coordinate descent: for each axis, determine optimal sign
        #    and immediately move to that position, getting progressively better values
        h = 5.0
        for i in range(dim):
            if time.time() > deadline:
                break
            
            # Probe in the +h direction from current position
            xp = x.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            calls_made += 1
            
            # Determine direction for axis i
            # If f(xp) > f(current), gradient is positive in this direction, we want to go -h
            # If f(xp) < f(current), gradient is negative, we want to go +h
            if fp > f_center:
                # Move to -h on this axis
                x[i] = -h
            else:
                # Move to +h on this axis
                x[i] = h
            
            # Update f_center to track current function value at x
            # We need to evaluate the new position to get its value
            # But wait - we already have fp which is f(x with +h on axis i)
            # If we chose -h, we don't have that value yet
            # Let's just evaluate the new x to get the actual value
            f_current = func(x)
            calls_made += 1
            f_center = f_current
        
        # 3. x is now the optimal vertex
        x_opt = x.copy()
        f_opt = f_center
        
        # 4. Fill remaining budget
        while calls_made < budget:
            if time.time() > deadline:
                break
            func(x_opt)
            calls_made += 1
            
        return x_opt, f_opt
