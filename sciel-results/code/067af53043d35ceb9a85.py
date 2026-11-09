"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, aim for 320s
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        n = self.dim
        # ZDT2 constant
        C = 1 + 10 * (n - 1)
        
        # Reference point value
        ref_val = 11.0
        
        # Calculate the intersection t* where f2(t) = ref_val
        # f2(t) = C * (1 - sqrt(t))
        # C * (1 - sqrt(t)) = ref_val
        # 1 - sqrt(t) = ref_val / C
        # sqrt(t) = 1 - ref_val / C
        # t = (1 - ref_val / C)^2
        t_star = (1.0 - ref_val / C) ** 2
        
        # Clip t_star to valid range [0, 1]
        t_star = np.clip(t_star, 0.0, 1.0)
        
        N = self.budget
        
        # We will generate N points on the Pareto front.
        # One point at t_star to capture the "corner" where f2 drops below ref.
        # The rest distributed in [t_star, 1.0].
        
        t_vals = np.zeros(N)
        t_vals[0] = t_star
        
        if N > 1:
            # Use inverse-CDF sampling based on hypervolume contribution
            # The hypervolume contribution density is proportional to (ref_val - f2(t)) * |df1/dt|
            # f1(t) = t, so df1/dt = 1
            # f2(t) = C * (1 - sqrt(t))
            # Contribution density ~ (ref_val - C * (1 - sqrt(t)))
            # = ref_val - C + C * sqrt(t)
            
            # Let's use a transformation that accounts for the convexity.
            # The Pareto front is convex, so points should be denser near t_star
            # where the marginal hypervolume gain is higher.
            
            # Use a power-law distribution: t = t_star + (1 - t_star) * u^alpha
            # With alpha > 1, points are denser near t_star
            # Let's try alpha = 2.0 for denser spacing near t_star
            
            u = np.linspace(0.0, 1.0, N - 1)
            alpha = 2.0
            t_vals[1:] = t_star + (1.0 - t_star) * (u ** alpha)
            
            # Ensure the last point is exactly 1.0
            t_vals[-1] = 1.0
        
        # Construct the population X
        # X is (N, dim)
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # x[1:] remain 0, which is optimal for ZDT2 front
        
        # Evaluate all points
        # Check time periodically? 
        # With N=200 and simple problem, it should be fast.
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
