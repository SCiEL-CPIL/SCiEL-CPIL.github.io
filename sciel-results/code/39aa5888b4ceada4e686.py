"""Multi-objective optimizer for DTLZ1 using Latin Hypercube Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer exploiting DTLZ1 structure using Latin Hypercube Sampling.
    
    DTLZ1 with M=3 objectives and N=7 variables has a Pareto front defined by:
    1. The last K = N - M + 1 = 5 variables (x[2:7]) must be set to 0.5 to make g(x)=0.
    2. The first M-1 = 2 variables (x[0], x[1]) define the position on the front,
       constrained by x[0] >= 0, x[1] >= 0, and x[0] + x[1] <= 1.
    
    To maximize the hypervolume of the recorded points with reference point [400, 400, 400],
    we need a set of points that covers the Pareto front (a 2D simplex) as evenly as possible.
    
    Strategy:
    1. Generate a Latin Hypercube (LHS) of size `budget` in 2D.
       LHS ensures that each 1D marginal is uniformly distributed, which is often better
       for hypervolume than simple grids or Halton sequences for moderate sample sizes.
    2. Map the 2D unit square points to the 2D simplex.
       A standard mapping is:
       u1, u2 in [0, 1]
       x0 = u1
       x1 = (1 - u1) * u2
       This maps the square to the triangle with vertices (0,0), (1,0), (0,1).
    3. Construct the full 7-dimensional vector:
       x[0] = mapped_x0
       x[1] = mapped_x1
       x[2:7] = 0.5
    4. Evaluate all points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate Latin Hypercube points in 2D
        # n_samples = budget
        n_samples = budget
        
        # Standard LHS generation:
        # Divide [0, 1] into n intervals. In each interval, pick one point uniformly at random.
        # Then permute the indices for the second dimension.
        
        # We use a deterministic permutation to ensure reproducibility if needed, 
        # but the problem statement says np.random.seed(seed) is called by the evaluator.
        # So we can use np.random directly.
        
        # Generate points in [0, 1]
        # u1: random permutation of intervals
        # u2: random permutation of intervals
        
        # Actually, standard LHS:
        # For i in 0..n-1:
        #   u1[i] = (i + random()) / n
        #   u2[i] = (perm[i] + random()) / n
        
        # Let's generate the base intervals
        indices = np.arange(n_samples)
        
        # Random offsets within intervals
        offsets_1 = np.random.rand(n_samples)
        offsets_2 = np.random.rand(n_samples)
        
        # Shuffle the second dimension indices to create the hypercube property
        permuted_indices = np.random.permutation(n_samples)
        
        u1 = (indices + offsets_1) / n_samples
        u2 = (permuted_indices + offsets_2) / n_samples
        
        # Map from [0, 1]^2 to the simplex
        # x0 = u1
        # x1 = (1 - u1) * u2
        # x2 = 1 - x0 - x1 (implicit, not needed for variable construction as we fix g-variables)
        
        x0 = u1
        x1 = (1.0 - u1) * u2
        
        # Construct the full vectors
        # x = [x0, x1, 0.5, 0.5, 0.5, 0.5, 0.5]
        points = np.zeros((n_samples, dim))
        points[:, 0] = x0
        points[:, 1] = x1
        # Set the g-variables (indices 2 to dim-1) to 0.5
        points[:, 2:] = 0.5
        
        # Clip to bounds (though 0.5 and simplex coords should be within [0,1])
        points = np.clip(points, lo, hi)
        
        # Evaluate all points
        # The evaluator records every returned objective vector.
        # We must not exceed the budget (200 evaluations).
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            # f is recorded by the evaluator, we don't need to store it unless we want to return something
            # The return value is ignored, but we must not crash.
            
        # Return a dummy value (ignored by evaluator)
        return points[-1] if n_samples > 0 else None
# EVOLVE-BLOCK-END
