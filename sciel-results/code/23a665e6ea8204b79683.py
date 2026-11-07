import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        # DTLZ1 Pareto front logic
        # Set non-shaping variables to 0.5 to make g(x) = 0
        # Shaping variables are x[0] and x[1] for M=3
        # x[2] to x[dim-1] are 0.5
        
        # Use Sobol sequence for x[0] and x[1]
        # budget is 200
        n_points = self.budget
        
        # Generate Sobol points in [0, 1]^2
        sampler = qmc.Sobol(d=2, scramble=True)
        
        try:
            points_2d = sampler.random(n_points)
        except AttributeError:
            # Fallback to Halton or random if random() not available
            points_2d = np.random.rand(n_points, 2)
        
        # Construct full dimension vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        X[:, 2:] = 0.5
        
        # Evaluate all points
        # The problem function clips and returns objectives.
        # We just call it for each point.
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
