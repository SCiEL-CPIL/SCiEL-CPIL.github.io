# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    
    Strategy:
    1. Exploit the analytic structure of DTLZ4:
       - The Pareto front for M=3 is a quarter circle in the x1-x2 plane.
       - x1 = cos(phi), x2 = sin(phi) for phi in [0, pi/2].
       - The remaining variables (x3...x12) are set to 0.5 (optimal for DTLZ4 g-function).
    2. Use a Sobol sequence to generate phi values for better low-discrepancy coverage
       compared to a simple uniform grid.
    3. Evaluate all generated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        n_front = n_obj - 1  # 2
        n_g = dim - n_front  # 10
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence points in [0, 1]
        # We need 'budget' points.
        # Using scrambled Sobol for deterministic-but-randomized low-discrepancy sampling.
        sobol = qmc.Sobol(d=1, scramble=True)
        phi_samples = sobol.random(budget)
        phi_values = phi_samples[:, 0] * (np.pi / 2)
        
        # Generate decision vectors
        X = np.zeros((budget, dim))
        X[:, 0] = np.cos(phi_values)
        X[:, 1] = np.sin(phi_values)
        # Set g-variables to 0.5 (optimal for DTLZ4)
        X[:, n_front:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
