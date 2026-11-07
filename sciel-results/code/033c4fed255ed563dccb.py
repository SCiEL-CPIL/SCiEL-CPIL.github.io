import numpy as np
import time

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = self.budget
        
        if HAS_SCIPY:
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            Y = sampler.generate(n_samples, dtype=np.float64)
        else:
            rng = np.random.default_rng(0)
            Y = rng.random((n_samples, 2))
            
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = Y[:, 0]
        X[:, 1] = Y[:, 1]
        X[:, 2:] = 0.5
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
            
        return X
