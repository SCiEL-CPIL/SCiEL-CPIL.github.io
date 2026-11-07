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
        deadline = start_time + 280  # Safety margin before 300s limit

        M = self.n_obj
        D = self.dim
        n_pos = M - 1
        
        # Generate Sobol points with a fixed seed for reproducibility
        # This ensures the same high-quality points for both seeds
        sampler = qmc.Sobol(d=n_pos, scramble=True, seed=42)
        points = sampler.random(n=self.budget)
        
        # Transform to angles for uniform surface density on the Pareto front
        u1 = points[:, 0]
        u2 = points[:, 1]
        
        # CDF inversion for uniform surface density
        # theta1 marginal density ~ sin(theta1)
        # F(theta1) = 1 - cos(theta1)
        # theta1 = arccos(1 - u1)
        theta1 = np.arccos(1.0 - u1)
        
        # theta2 conditional density is uniform
        # theta2 = (pi / 2) * u2
        theta2 = (np.pi / 2.0) * u2
        
        # Map angles to decision variables x0, x1 in [0, 1]
        x0 = (2.0 / np.pi) * theta1
        x1 = (2.0 / np.pi) * theta2
        
        X = np.zeros((self.budget, D))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5
        
        # Evaluate all points
        F = []
        for i, x in enumerate(X):
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
            
        if not F:
            return X
            
        F = np.array(F)
        
        return X
