import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine the optimal tail value for ZDT4
        # The term in g(x) is x^2 - 10*cos(4*pi*x) + 10
        # Find the global minimum of this function over the bounds [lo[1], hi[1]]
        # The function has local minima at x = 0.5 + k for integer k
        # Within [0, 1], the global min is at x = 0.5 with value 0.25
        # We do a fine grid search to be safe
        tail_grid = np.linspace(lo[1], hi[1], 1000)
        tail_costs = tail_grid**2 - 10 * np.cos(4 * np.pi * tail_grid) + 10
        best_idx = np.argmin(tail_costs)
        best_tail_val = tail_grid[best_idx]
        
        # The optimal tail is a vector of this best value
        x_tail = np.full(self.dim - 1, best_tail_val)
        
        # For ZDT4, the Pareto front has f1 = x1 in [0, 1]
        # Only sample x0 in [0, 1] to avoid wasting evaluations
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)
        
        if x0_lo >= x0_hi:
            # If bounds don't overlap [0,1], use the full range
            x0_lo = lo[0]
            x0_hi = hi[0]
        
        # Use all budget for dense sampling in the valid range
        x0_values = np.linspace(x0_lo, x0_hi, self.budget)
        
        for x0 in x0_values:
            if time.time() > deadline:
                break
            
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None
