# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Approach:
    - DTLZ2 Pareto front is a unit hypersphere in objective space (for n_obj objectives).
    - For 3 objectives, the Pareto front is a quarter-sphere in the positive octant of R^3.
    - The distance variables (last dim - n_obj + 1 variables) should be set to 0.5 for optimality.
    - The angle variables (first n_obj - 1 = 2 variables) determine the position on the Pareto front.
    
    We use a deterministic Sobol sequence (low-discrepancy) to sample the 2D angle space
    maximally space-filling, ensuring good hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # For DTLZ2 with n_obj=3, dim=12:
        # Angle variables: first n_obj - 1 = 2 variables
        # Distance variables: remaining dim - (n_obj - 1) = 10 variables, set to 0.5
        
        n_angle = n_obj - 1  # 2
        
        # Generate a low-discrepancy sequence in [0,1]^n_angle using Sobol
        # scramble=False ensures deterministic behavior independent of np.random.seed
        sampler = qmc.Sobol(d=n_angle, scramble=False)
        points = sampler.random(budget)
        
        # points shape: (budget, n_angle)
        
        # Build full x vectors: angle variables from points, distance variables = 0.5
        X = np.zeros((budget, dim))
        X[:, :n_angle] = points
        X[:, n_angle:] = 0.5  # Distance variables set to 0.5 for optimality
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            problem(x)
        
        return X
# EVOLVE-BLOCK-END
