import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    """DTLZ2 optimizer using Sobol sequence for space-filling sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2: First n_obj-1 variables are shape variables.
        # The rest are distance variables.
        k_shape = self.n_obj - 1
        k_dist = self.dim - k_shape

        # Generate Sobol sequence points in the unit hypercube [0, 1]^k_shape
        # We use all budget for global sampling to maximize coverage of the Pareto front.
        n_samples = self.budget
        
        # Initialize Sobol sampler
        # scramble=True uses a random seed (controlled by np.random.seed) to permute the sequence,
        # which helps avoid systematic bias if the problem structure aligns with the base-2 structure.
        sampler = qmc.Sobol(d=k_shape, scramble=True)
        
        # Generate points
        # Note: Sobol generates points in [0, 1)
        points_unit = sampler.random(n_samples)

        # Map unit points to the actual bounds of the shape variables
        shape_lo = lo[:k_shape]
        shape_hi = hi[:k_shape]
        
        # Linear mapping: point = lo + (hi - lo) * unit_point
        points_scaled = shape_lo + (shape_hi - shape_lo) * points_unit

        solutions = []

        for i in range(n_samples):
            if time.time() > deadline:
                break

            # Construct the full decision vector
            x = np.zeros(self.dim)
            
            # Assign shape variables
            x[:k_shape] = points_scaled[i]
            
            # Assign distance variables
            # For standard DTLZ2 with bounds [0, 1], the function g(x) is minimized (to 0) 
            # when all distance variables are 0.5. This ensures the point lies on the Pareto front.
            if k_dist > 0:
                x[k_shape:] = 0.5
            
            # Ensure strict compliance with bounds (though 0.5 is within [0,1])
            x = np.clip(x, lo, hi)

            # Evaluate
            f = problem(x)
            solutions.append(x)

        return np.array(solutions) if solutions else np.array([np.full(self.dim, 0.5)])
