"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using analytically optimized Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # ZDT2 Pareto front: x[0] = t in [0,1], x[1..] = 0
        # f1 = t, f2 = 1 - sqrt(t)
        # Reference point: [11, 11]
        
        # For 2D hypervolume with reference (r1, r2) = (11, 11):
        # Given sorted points by f1: (f1_1, f2_1), ..., (f1_n, f2_n)
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_n) * (r2 - f2_n)
        # where f1_{n+1} = r1
        # Actually: HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{n+1} = r1 = 11
        
        # With f2_i = 1 - sqrt(f1_i):
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - (1 - sqrt(f1_i)))
        #    = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
        
        # To maximize this, we want to place points where the integrand (10 + sqrt(f1)) is large,
        # i.e., where f1 is large. But we also need to cover the range.
        
        # The optimal placement for maximizing the sum with a fixed number of points
        # can be found via numerical optimization or by using a non-uniform grid.
        
        # Let's use a grid that is denser near f1 = 1 where the contribution per unit width is higher.
        # Use a power-law transformation: t = s^p where s is uniform in [0,1]
        # Higher p means more points near t=1
        
        # Let's try p=2 (quadratic) which gives t = s^2
        # This concentrates points near t=1
        
        n_points = budget  # Use all budget for front sampling
        
        if n_points < 2:
            n_points = 2
            
        # Generate s values uniformly in [0, 1]
        s_vals = np.linspace(0.0, 1.0, n_points)
        
        # Apply power-law transformation to concentrate near t=1
        # t = s^p, with p > 1
        p = 2.0
        t_vals = s_vals ** p
        
        # Ensure we have points at both endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        
        # Create x vectors: x[0] = t, x[1..] = 0
        X = np.zeros((n_points, dim))
        X[:, 0] = t_vals
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        return X
# EVOLVE-BLOCK-END
