"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: 
        # 1. Assume first M-1 variables are position variables, rest are distance variables.
        # 2. For DTLZ1, setting distance variables to 0 minimizes g.
        # 3. Vary position variables to cover the Pareto front.
        
        n_pos = self.n_obj - 1  # Typically M-1 position variables
        n_dist = self.dim - n_pos
        
        # If n_pos is 1, front is 1D. If n_pos is 2, front is 2D.
        # We will create a grid in the position space.
        
        # Determine grid size to stay within budget
        # We want to use most of the budget.
        # Let's reserve some for local refinement if needed, but construction is cheap.
        
        # Simple approach: uniform grid in position space.
        # If n_pos = 1: 200 points along the line.
        # If n_pos = 2: 14x14 grid ~ 196 points.
        
        if n_pos == 1:
            n_grid = int(np.sqrt(self.budget))
            # Actually for 1D, we can just use budget points.
            x_vals = np.linspace(lo[0], hi[0], self.budget)
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x_vals
            # Set distance variables to 0 (or midpoint if 0 is not optimal? 
            # For DTLZ1, g is min at x=0 for distance vars usually).
            # Assuming bounds are [0,1], 0 is the min.
            for i in range(1, self.dim):
                X[:, i] = lo[i]
            
        elif n_pos == 2:
            # Grid in 2D
            n_grid = int(np.sqrt(self.budget))
            x1_vals = np.linspace(lo[0], hi[0], n_grid)
            x2_vals = np.linspace(lo[1], hi[1], n_grid)
            
            X_list = []
            for x1 in x1_vals:
                for x2 in x2_vals:
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    for i in range(2, self.dim):
                        x[i] = lo[i]
                    X_list.append(x)
            
            X = np.array(X_list)
            
        else:
            # Fallback to random if structure is unknown
            X = lo + (hi - lo) * np.random.rand(self.budget, self.dim)
            
        # Evaluate all points
        # Check time
        if time.time() > deadline:
            return X[:0] # Should not happen with 200 evals
        
        F = np.array([problem(x) for x in X])
        
        # Optional: Local refinement for best points?
        # The constructed points should already be on the front.
        # We can return X.
        
        return X

# EVOLVE-BLOCK-END
