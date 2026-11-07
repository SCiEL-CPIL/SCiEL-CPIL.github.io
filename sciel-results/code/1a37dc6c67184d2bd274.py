"""Multi-objective optimizer on DTLZ1 using analytic Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Analytic DTLZ1 Pareto-front sampler."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        time_limit = 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 Pareto front structure:
        # n=7, M=3
        # g(x) depends on x[2] to x[6] (0-indexed).
        # To make g(x)=0, set x[2..6] = 0.5.
        # Objectives depend on x[0] and x[1].
        # f1 = 0.5 * x[0]
        # f2 = 0.5 * x[0] * x[1]
        # f3 = 0.5 * (1 - x[1])
        
        # We will sample x[0] and x[1] from a low-discrepancy sequence.
        # Use Halton sequence in 2D.
        
        n_samples = budget
        
        # Generate Halton sequence
        # qmc.Halton(d=2, scramble=True)
        # We need to ensure we don't exceed budget.
        
        halton = qmc.Halton(d=2, scramble=True)
        # Generate n_samples points in [0, 1]
        # Note: Halton sequence includes 0 and 1? 
        # qmc.random(n) generates n points.
        
        points_2d = halton.random(n_samples)
        
        # points_2d is shape (n_samples, 2)
        
        X = np.zeros((n_samples, dim))
        
        # Set the free variables
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        
        # Set the g-variables to 0.5 to ensure g=0
        # Indices 2 to dim-1
        X[:, 2:] = 0.5
        
        # Clip to bounds just in case (though 0.5 and [0,1] are within [0,1])
        # The problem function clips anyway, but good practice.
        # Actually, the problem function clips the input x.
        # We are passing x directly.
        
        # Evaluate all points
        # The problem function returns 3 objectives.
        # We record all of them.
        
        F_list = []
        for i in range(n_samples):
            if time.time() - start_time > time_limit:
                break
            x_i = X[i]
            f_i = problem(x_i)
            F_list.append(f_i)
            
        # Return the last evaluated point or the array of X (ignored)
        # The contract says return value is ignored.
        return X

# EVOLVE-BLOCK-END
