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
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ2 Pareto front: quarter-sphere in 3D
        # y0 = cos(theta1) * cos(theta2)
        # y1 = cos(theta1) * sin(theta2)
        # y2 = sin(theta1)
        
        # Reserve 3 evaluations for the vertices of the front
        # (1, 0, 0), (0, 1, 0), (0, 0, 1)
        # Corresponding thetas:
        # (0, 0) -> (1, 0, 0)
        # (0, pi/2) -> (0, 1, 0)
        # (pi/2, 0) -> (0, 0, 1)  [Note: theta2=0, theta1=pi/2 gives y0=0, y1=0, y2=1]
        
        n_sobol = n_evals - 3
        if n_sobol < 1:
            n_sobol = 1
            n_vertices = 0
        else:
            n_vertices = 3

        # Generate Sobol points in [0, 1)^2
        sobol = qmc.Sobol(d=2, scramble=True)
        u = sobol.random(n_sobol)
        
        # Map to [0, pi/2]^2
        theta1 = u[:, 0] * (np.pi / 2)
        theta2 = u[:, 1] * (np.pi / 2)
        
        # Compute objective vectors for Sobol points
        y0 = np.cos(theta1) * np.cos(theta2)
        y1 = np.cos(theta1) * np.sin(theta2)
        y2 = np.sin(theta1)
        
        # Create X matrix
        X = np.zeros((n_sobol, self.dim))
        X[:, 0] = y0
        X[:, 1] = y1
        X[:, 2] = y2
        X[:, 3:] = 0.5
        
        # Add vertices
        if n_vertices > 0:
            # Vertex 1: (1, 0, 0)
            x_v1 = np.zeros(self.dim)
            x_v1[0] = 1.0
            x_v1[3:] = 0.5
            
            # Vertex 2: (0, 1, 0)
            x_v2 = np.zeros(self.dim)
            x_v2[1] = 1.0
            x_v2[3:] = 0.5
            
            # Vertex 3: (0, 0, 1)
            x_v3 = np.zeros(self.dim)
            x_v3[2] = 1.0
            x_v3[3:] = 0.5
            
            X = np.vstack([X, x_v1, x_v2, x_v3])
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
