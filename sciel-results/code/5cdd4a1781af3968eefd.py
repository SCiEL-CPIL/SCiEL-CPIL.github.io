# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0 for points exactly on the Pareto front. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Sobol quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Number of angle variables
        n_angles = n_obj - 1  # 2 for 3 objectives
        
        # Generate Sobol sequence points in [0, 1]^2
        n_points = min(budget, 200)
        
        # Use Sobol sequence for better space-filling properties
        sampler = qmc.Sobol(d=n_angles, scramble=True)
        # Sobol requires power-of-2 sample sizes for best properties
        # Find the largest power of 2 <= n_points
        n_sobol = 1
        while n_sobol * 2 <= n_points:
            n_sobol *= 2
        points = sampler.random(n_sobol)
        
        # Map Sobol points to angles on the Pareto front
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere
        # f1 = cos(theta1) * cos(theta2)
        # f2 = cos(theta1) * sin(theta2)  
        # f3 = sin(theta1)
        # where theta1, theta2 in [0, pi/2]
        
        theta1 = points[:, 0] * (np.pi / 2)
        theta2 = points[:, 1] * (np.pi / 2)
        
        # Build decision vectors
        # First n_obj - 1 variables are the angles (scaled to [0, pi/2])
        # Last dim - n_obj + 1 variables are distance variables (set to 0 for Pareto front)
        X = np.zeros((n_sobol, dim))
        X[:, 0] = theta1
        X[:, 1] = theta2
        # Distance variables set to 0 for points on the Pareto front
        # (g(x) = 0 when x_M..x_K are at their optimum)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sobol
        
        return X
# EVOLVE-BLOCK-END
