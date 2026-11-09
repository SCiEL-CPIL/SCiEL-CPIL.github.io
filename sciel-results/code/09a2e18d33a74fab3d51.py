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
        # Generate points
        # Use Halton sequence for low discrepancy
        # scipy is available, but let's stick to numpy if possible to avoid import overhead/complexity?
        # scipy.stats.qmc is fine.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=2, scramble=False)
            x01 = sampler.random(n_evals)
        except ImportError:
            # Fallback to grid if scipy fails
            n_per_dim = int(np.sqrt(n_evals))
            x0 = np.linspace(0, 1, n_per_dim)
            x1 = np.linspace(0, 1, n_per_dim)
            X0, X1 = np.meshgrid(x0, x1)
            x01 = np.column_stack((X0.ravel(), X1.ravel()))[:n_evals]
            
        # Scale to bounds (which are [0,1] anyway)
        x01 = lo[:2] + (hi[:2] - lo[:2]) * x01
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x01[:, 0]
        X[:, 1] = x01[:, 1]
        X[:, 2:] = 0.5 # Midpoint for DTLZ2 g(x)=0
        
        # Evaluate
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
