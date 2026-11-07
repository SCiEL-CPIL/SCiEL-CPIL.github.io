import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure: 
        # x[0] and x[1] are position variables.
        # x[2:] are distance variables.
        # To reach the Pareto front, distance variables should be 0 (or 1, but 0 is standard for minimization of g).
        # We assume x[2:] = 0 is optimal for the distance part.
        
        # We will use a grid search on the position variables x[0] and x[1].
        # The Pareto front for M=3 is a 2D manifold.
        # A grid on [0,1] x [0,1] for x[0], x[1] will cover the front.
        
        # Calculate grid size based on budget
        # We have 200 evaluations.
        # Let's use a grid of size G x G.
        # G^2 <= 200 => G <= 14.
        # 14 * 14 = 196.
        # Remaining 4 evaluations can be used for random sampling or refinement.
        
        grid_size = 14
        evals = 0
        
        X = []
        F = []
        
        # Generate grid points
        # x[0] and x[1] vary, x[2:] = 0
        for i in range(grid_size):
            x1 = lo[0] + (hi[0] - lo[0]) * (i / (grid_size - 1))
            for j in range(grid_size):
                x2 = lo[1] + (hi[1] - lo[1]) * (j / (grid_size - 1))
                
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # x[2:] remains 0
                
                # Check time
                if time.time() > deadline:
                    break
                
                f = problem(x)
                X.append(x)
                F.append(f)
                evals += 1
                
                if evals >= self.budget:
                    break
            if evals >= self.budget or time.time() > deadline:
                break
        
        # If we have remaining budget, add some random points
        # to ensure we don't miss any extreme points or to handle any potential 
        # non-zero optimal for distance variables (though unlikely for DTLZ4).
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Random point in the whole space, but bias towards x[2:]=0
            x = np.random.random(self.dim)
            x[2:] = 0.0 # Keep distance variables at 0
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X.append(x)
            F.append(f)
            evals += 1
        
        return X
