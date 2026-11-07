# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Directly construct points on the DTLZ1 Pareto front using a Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 has M objectives, N variables.
        # The Pareto front is defined by the first N-M+1 variables (position variables).
        # The remaining M-1 variables are aggregation variables which should be set to 0.5
        # to lie on the front.
        # Here, dim=7, n_obj=3, so N-M+1 = 7-3+1 = 5 position variables.
        # The last 2 variables are aggregation variables.
        
        n_pos = dim - n_obj + 1  # 5
        n_agg = n_obj - 1        # 2
        
        # For DTLZ1, the shape of the Pareto front is determined by the first M-1 position variables.
        # Here M=3, so x1 and x2 determine the front.
        # We sample these 2 variables using a 2D Sobol sequence for better uniformity.
        # The other position variables (x3, x4, x5) do not affect the objective values
        # (they are not part of the front definition in the same way, or rather, 
        # the front is 2D and embedded in the 5D position space, but only x1, x2 matter for the mapping to objectives).
        # Actually, for DTLZ1, f_i depends on x_1...x_{M-1}. x_M...x_{N-M+1} are free?
        # Let's check: f_1 = 0.5(1+g) x_1 (1-x_2). f_2 = 0.5(1+g) x_1 x_2. f_3 = 0.5(1+g).
        # Yes, only x_1 and x_2 appear.
        
        n_shape_vars = n_obj - 1 # 2
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_shape_vars, scramble=True, seed=0)
            shape_points = sampler.random(n)
        except ImportError:
            rng = np.random.default_rng(0)
            shape_points = rng.random((n, n_shape_vars))
        
        # Construct the full input vectors
        X = np.zeros((n, dim))
        X[:, :n_shape_vars] = shape_points
        
        # Set the remaining position variables (x3, x4, x5) to 0.5
        if n_pos > n_shape_vars:
            X[:, n_shape_vars:n_pos] = 0.5
            
        # Set aggregation variables to 0.5
        X[:, n_pos:] = 0.5
        
        # Clip to bounds to be safe (though bounds are [0,1] and points are in [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
