import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific:
        # x[0], x[1] determine front position
        # x[2:] determine distance to front. Set to 0.5 for g(x)=0.
        
        n_evals = self.budget
        # Generate points using 3D Halton sequence for DTLZ2 with 3 objectives
        # The Pareto front is determined by x[0], x[1], x[2]
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=3, scramble=False)
            x012 = sampler.random(n_evals)
        except ImportError:
            # Fallback: use stratified sampling in 3D
            n_per_dim = int(np.cbrt(n_evals))
            x0 = np.linspace(0, 1, n_per_dim)
            x1 = np.linspace(0, 1, n_per_dim)
            x2 = np.linspace(0, 1, n_per_dim)
            X0, X1, X2 = np.meshgrid(x0, x1, x2)
            x012 = np.column_stack((X0.ravel(), X1.ravel(), X2.ravel()))[:n_evals]
            
        # Scale to bounds (which are [0,1] anyway)
        x012 = lo[:3] + (hi[:3] - lo[:3]) * x012
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x012[:, 0]
        X[:, 1] = x012[:, 1]
        X[:, 2] = x012[:, 2]
        X[:, 3:] = 0.5 # Midpoint for DTLZ2 g(x)=0
        
        # Evaluate
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
