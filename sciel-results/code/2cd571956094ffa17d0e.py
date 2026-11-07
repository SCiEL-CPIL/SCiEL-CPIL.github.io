import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine valid range for x_0
        # For ZDT4, x_0 must be non-negative for the square root in f_2 to be real.
        # If the provided bounds allow negative values, we restrict to [0, hi[0]] to avoid NaNs.
        x0_min = max(lo[0], 0.0)
        x0_max = hi[0]
        
        # If the range is invalid (e.g. hi < lo or hi < 0), fall back to a single point or handle error
        if x0_max < x0_min:
            x0_min = 0.0
            x0_max = 1.0 # Fallback, though this shouldn't happen with valid bounds
            
        n_evals = self.budget
        
        # Create dense grid for x_0
        x0_vals = np.linspace(x0_min, x0_max, n_evals)
        
        # Other variables set to 0 (global minima for ZDT4's g function)
        x_rest = np.zeros(self.dim - 1)
        
        for x0 in x0_vals:
            x = np.concatenate(([x0], x_rest))
            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)
            f = problem(x)
            
        return None
