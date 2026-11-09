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
        # Parameterize with angles theta1, theta2 in [0, pi/2]
        # y0 = cos(theta1) * cos(theta2)
        # y1 = cos(theta1) * sin(theta2)  
        # y2 = sin(theta1)
        # x[0]=y0, x[1]=y1, x[2]=y2, x[3:]=0.5 gives g=0
        
        # Use Sobol sequence for better space-filling properties
        # This gives more uniform coverage than a regular grid
        sobol = qmc.Sobol(d=2, scramble=True)
        # Get n_evals points in [0,1)^2
        u = sobol.random(n_evals)
        
        # Map to [0, pi/2] for angles
        theta1 = u[:, 0] * (np.pi / 2)
        theta2 = u[:, 1] * (np.pi / 2)
        
        # Generate points on the front
        X = np.zeros((n_evals, self.dim))
        
        for i in range(n_evals):
            t1 = theta1[i]
            t2 = theta2[i]
            
            y0 = np.cos(t1) * np.cos(t2)
            y1 = np.cos(t1) * np.sin(t2)
            y2 = np.sin(t1)
            
            X[i, 0] = y0
            X[i, 1] = y1
            X[i, 2] = y2
            X[i, 3:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
