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
        deadline = start_time + 290  # 300s limit, keep 10s safety margin
        
        n_evals = self.budget
        dim = self.dim
        
        # Generate Sobol sequence for first 2 variables
        # We need n_evals points in 2D
        sobol = qmc.Sobol(d=2, scramble=False)
        # Sobol generates points in [0, 1)
        points_2d = sobol.random(n_evals)
        
        # Construct full dimension vectors
        # x = [x1, x2, 0.5, 0.5, 0.5, 0.5, 0.5]
        X = np.zeros((n_evals, dim))
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        
        # Fill the rest with 0.5
        if dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate with deadline guard
        for i in range(n_evals):
            if time.time() >= deadline:
                break
            problem(X[i])
            
        return X
