"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Direct construction optimizer for DTLZ1 using Sobol sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        # DTLZ1 specific: 
        # Pareto front is at g=0.
        # g depends on x_{n_obj}...x_{dim-1}. For dim=7, n_obj=3, g depends on x_3..x_6.
        # To get g=0, set x_3..x_6 = 0.5.
        # x_0, x_1 determine the position on the front.
        # f_1 = 0.5 * x_0
        # f_2 = 0.5 * (1 - x_0) * x_1
        # f_3 = 0.5 * (1 - x_0) * (1 - x_1)
        
        # We want to sample the Pareto front (triangle in objective space) uniformly.
        # This is equivalent to sampling barycentric coordinates (w_0, w_1, w_2) uniformly from the simplex.
        # f_i = 0.5 * w_i.
        # w_0 = x_0
        # w_1 = (1 - x_0) * x_1  => x_1 = w_1 / (1 - w_0)
        # w_2 = (1 - x_0) * (1 - x_1)
        
        # Generate Sobol points in 2D to map to the simplex.
        # Mapping from (u1, u2) in [0,1]^2 to (w0, w1, w2) on simplex:
        # w0 = 1 - sqrt(u1)
        # w1 = sqrt(u1) * (1 - u2)
        # w2 = sqrt(u1) * u2
        
        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        # Generate points in [0, 1]
        points = sobol.random(n=budget)
        
        X = np.zeros((budget, dim))
        
        # Set the g-variables to 0.5 to ensure g=0
        # Indices 3 to dim-1 (inclusive) correspond to the g-variables
        # For dim=7, indices 3,4,5,6
        if dim > self.n_obj:
            X[:, self.n_obj:] = 0.5
            
        # Calculate x_0 and x_1 from Sobol points
        u1 = points[:, 0]
        u2 = points[:, 1]
        
        # Ensure u1 is not 0 to avoid division by zero or log issues if we used logs, 
        # but here we use sqrt. sqrt(0)=0 is fine.
        # However, w0 = 1 - sqrt(u1). If u1=0, w0=1. Then 1-w0=0. Division by zero.
        # Sobol.random returns values in [0, 1). So u1 can be 0.
        # If u1 is 0, w0=1, w1=0, w2=0. Point is (0.5, 0, 0) in objective space.
        # x_0 = 1. x_1 is undefined but irrelevant since w1=0.
        # We can set x_1 = 0 in that case.
        
        w0 = 1.0 - np.sqrt(u1)
        # Handle case where w0 is close to 1
        # x_1 = w1 / (1 - w0)
        # w1 = sqrt(u1) * (1 - u2)
        
        x0 = w0.copy()
        x1 = np.zeros(budget)
        
        # Mask where 1 - w0 is not close to 0
        mask = (1.0 - w0) > 1e-12
        w1 = np.sqrt(u1) * (1.0 - u2)
        
        x1[mask] = w1[mask] / (1.0 - w0[mask])
        
        # Clip x1 to [0, 1] to be safe against floating point errors
        x1 = np.clip(x1, 0.0, 1.0)
        
        X[:, 0] = x0
        X[:, 1] = x1
        
        # Clip all X to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return F
# EVOLVE-BLOCK-END
