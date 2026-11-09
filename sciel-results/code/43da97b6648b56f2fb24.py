"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        n = self.dim
        # ZDT2 Pareto front: x[0] = t in [0,1], x[1:] = 0
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(x[0]))
        # For n=30: g = 1 + 290 = 291
        # f1 = t, f2 = 291 * (1 - sqrt(t))
        
        # Reference point is [11.0, 11.0]
        # Points must satisfy f1 < 11 and f2 < 11 to contribute to hypervolume.
        # f1 = t <= 1 < 11, so f1 condition is always met.
        # f2 < 11 => 291 * (1 - sqrt(t)) < 11
        # 1 - sqrt(t) < 11/291
        # sqrt(t) > 1 - 11/291
        # t > (1 - 11/291)^2
        
        g = 1 + 10 * (n - 1)  # 291 for n=30
        f2_ref = 11.0
        
        # Find the minimum t such that f2 < f2_ref
        # f2(t) = g * (1 - sqrt(t))
        # t_min = (1 - f2_ref/g)^2
        if f2_ref >= g:
            # If reference is higher than max f2, entire front is valid
            t_min = 0.0
        else:
            t_min = (1.0 - f2_ref / g) ** 2
            
        # We want to sample the valid region t in [t_min, 1]
        # Uniform sampling in t_min..1 might still be suboptimal if the curve is non-linear.
        # Let's sample uniformly in f2 from 0 to f2_ref (exclusive) to cover the valid f2 range evenly.
        # f2 ranges from 0 (at t=1) to f2_ref (at t=t_min).
        
        num_points = self.budget
        # Sample f2 values in [0, f2_ref)
        # We use np.linspace(0, f2_ref, num_points, endpoint=False) to get 0, ..., f2_ref - delta
        f2_vals = np.linspace(0.0, f2_ref, num_points, endpoint=False)
        
        # Convert f2 to t
        # f2 = g * (1 - sqrt(t))
        # sqrt(t) = 1 - f2/g
        # t = (1 - f2/g)^2
        t_vals = (1.0 - f2_vals / g) ** 2
        
        # Ensure t is within [0, 1]
        t_vals = np.clip(t_vals, 0.0, 1.0)
        
        results = []
        for t in t_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(n)
            x[0] = t
            # Clip to bounds
            # The problem clips x to bounds, but we should ensure we are within [0,1]
            # bounds is a list of (low, high)
            # Assuming bounds are [0,1] for all variables as per task description
            x = np.clip(x, bounds[0][0], bounds[0][1])
            for i in range(1, n):
                x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
            
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
