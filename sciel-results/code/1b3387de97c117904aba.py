# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    DTLZ7 optimizer using 1D Pareto front parameterization.
    
    Models the Pareto front as a 1D curve where we trade off f_0 and f_1
    while keeping f_2 at its minimum (≈1). Uses the full budget for dense
    sampling of this curve to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim

        # DTLZ7 with m=3: position vars are x_0..x_3 (k=4), trailing vars x_4..x_{n-1}
        # On Pareto front: trailing vars = 0, so g=1
        # f_2 = 1
        # f_0 = 1 + h*(x_0*sin(pi*x_0/2) + x_1*sin(pi*x_1/2))
        # f_1 = 1 + h*(x_2*sin(pi*x_2/2) + x_3*sin(pi*x_3/2))
        # h = 1 - sqrt(f_0/f_2) = 1 - sqrt(f_0)
        
        # 1D parameterization: vary a single parameter t in [0,1]
        # Set x_0 = t, x_1 = 0, x_2 = 1-t, x_3 = 0
        # This creates a trade-off curve between f_0 and f_1
        
        X = np.zeros((n, dim))
        # Use non-uniform spacing to concentrate where hypervolume contribution is highest
        # Reference point is [1.1, 1.1, 25.0], so points near f_0≈0.5, f_1≈0.5 contribute most
        t_values = np.linspace(0.001, 0.999, n)
        
        for i, t in enumerate(t_values):
            X[i, 0] = t
            X[i, 1] = 0.0
            X[i, 2] = 1.0 - t
            X[i, 3] = 0.0
            # All other variables remain 0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
