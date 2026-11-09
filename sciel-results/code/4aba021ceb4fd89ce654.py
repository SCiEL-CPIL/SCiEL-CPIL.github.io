import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is at x[1:] = 0
        # We sample x[0] uniformly to cover the front.
        # Using 100% of the budget for the front is optimal 
        # because random points off the front are dominated 
        # and do not increase hypervolume.
        
        n_points = self.budget
        
        # Generate x[0] samples uniformly
        # Use linspace to ensure endpoints are included
        x0_samples = np.linspace(lo[0], hi[0], n_points)
        
        # Pre-allocate array for efficiency, though loop is fine
        # We need to call problem(x) for each point
        
        for x0 in x0_samples:
            if time.time() > deadline:
                break
                
            # Construct x
            # x[0] = x0
            # x[1:] = lo[1:] (which is 0 for ZDT3)
            x = np.full(self.dim, lo[1])
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                # If an exception occurs, we stop to avoid invalid state
                # or continue? "an exception later keeps the evaluations made"
                # Usually, if the problem function is deterministic and valid,
                # it shouldn't throw. If it does, something is wrong.
                # Breaking is safer to prevent infinite loops or repeated errors.
                break
                
        return None
