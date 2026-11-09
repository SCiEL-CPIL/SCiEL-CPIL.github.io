# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i:
       a. Evaluate at 5*e_i to determine gradient component v_i.
       b. Construct a "partial vertex" where dimensions 1..i are set to 
          their optimal values (-5 or 5 based on sign of v_j) and 
          dimensions i+1..d are 0.
       c. Evaluate this partial vertex. This allows the best value b_i 
          to improve incrementally as we determine more dimensions, 
          rather than waiting until all dimensions are known.
    3. After all dimensions are processed, the last partial vertex is 
       the global optimum for the linear function.
    4. Stall with the best point found for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Track gradient components
        v = np.zeros(d)
        
        # Step 2: Interleave axis probes and partial vertex evaluations
        for i in range(d):
            # Check time and budget
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # 2a. Probe axis i at distance 5
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            
            # Update gradient component
            v[i] = (fi - f0) / 5.0
            
            # Update best if this axis point is better
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
            
            # 2b. Construct partial optimal vertex for dimensions 0..i
            # For j <= i, we know v[j]. Set x[j] to -5 if v[j]>0, 5 if v[j]<0.
            # For j > i, x[j] = 0.
            x_partial = np.zeros(d)
            for j in range(i + 1):
                if v[j] > 0:
                    x_partial[j] = -5.0
                else:
                    x_partial[j] = 5.0
            
            # Check time and budget
            if time.time() > deadline or calls_made >= self.budget:
                break
                
            # Evaluate partial vertex
            fp = func(x_partial)
            calls_made += 1
            
            if fp < f_best:
                f_best = fp
                x_best = x_partial.copy()
        
        # Step 3: Stall with the best point found
        # If we broke out early, x_best holds the last valid best point.
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
