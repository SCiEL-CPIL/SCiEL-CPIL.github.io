import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: 
        # WFG9 with 3 objectives and 12 variables.
        # Hypothesis: The first 3 variables (x0, x1, x2) define the Pareto front shape.
        # The remaining 9 variables (x3...x11) are "positional" or "noise" variables 
        # that can be fixed at midpoints to maximize the chance of being on or near 
        # the ideal front for the given x0, x1, x2.
        
        # Fix variables 3 to 11 at midpoints
        midpoints = (lo + hi) / 2.0
        base_x = midpoints.copy()
        
        # Use a dense 3D grid for the first 3 dimensions
        # 5 x 5 x 8 = 200 points
        n0, n1, n2 = 5, 5, 8
        grid0 = np.linspace(lo[0], hi[0], n0)
        grid1 = np.linspace(lo[1], hi[1], n1)
        grid2 = np.linspace(lo[2], hi[2], n2)
        
        count = 0
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    if time.time() > deadline:
                        return None
                    
                    x = base_x.copy()
                    x[0] = grid0[i0]
                    x[1] = grid1[i1]
                    x[2] = grid2[i2]
                    
                    problem(x)
                    count += 1
        
        return None
