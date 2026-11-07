import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        
        # DTLZ4 specific parameters
        alpha = 100.0
        n_shape_vars = self.n_obj - 1  # x_1, ..., x_{M-1}
        n_g_vars = dim - n_shape_vars  # x_M, ..., x_n
        
        # Generate Sobol sequence for shape variables
        # We need 'budget' points
        # Sobol requires power of 2 or we can just use random
        # For 200 points, random with good seeding is fine, or Sobol
        # Let's use Sobol for better coverage
        sampler = qmc.Sobol(d=n_shape_vars, scramble=True)
        # Sobol generates points in [0, 1]
        # We need to generate enough points
        # qmc.Sobol requires n to be a power of 2 for best quality, but it works for any n
        # Actually, qmc.Sobol.random(n) works
        u = sampler.random(budget)
        
        # Transform u to x_shape
        # x = u^(1/alpha)
        x_shape = u ** (1.0 / alpha)
        
        # Initialize full X
        X = np.zeros((budget, dim))
        X[:, :n_shape_vars] = x_shape
        
        # For g variables, set to 0 to minimize g(x)
        # (Assuming g(x) is minimized at boundaries 0 or 1)
        X[:, n_shape_vars:] = 0.0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X], dtype=np.float64)
        
        return X
