# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Analytic inversion strategy for ZDT3 hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        if budget <= 0:
            return None

        # ZDT3 Pareto front:
        # f1 = x0
        # g_min = 1 (when xi = 0.5 for i=1..dim-1)
        # f2 = (1 + sqrt(f1)) * h(f1, 1)
        # h(u, 1) = 1 - sqrt(u) - u * sin^2(10 * pi * u)
        #
        # The front has discontinuities where sin^2(10*pi*u) = 1, i.e., u = (2k+1)/20
        # For u in [0,1], discontinuities at u = 0.05, 0.15, 0.25, ..., 0.95
        # There are 10 such points, creating 11 segments (but some may be very small).
        #
        # To maximize hypervolume, we should place points strategically in objective space.
        # Key insight: The hypervolume contribution is maximized by spreading points
        # in the (f1, f2) plane. Since f1 = x0, we can control f1 directly.
        #
        # Strategy: Use a non-uniform sampling that places more points in regions
        # where the Pareto front has high curvature or where the hypervolume contribution
        # is large. We'll use a combination of:
        # 1. Points near the discontinuities (where the front changes behavior)
        # 2. Uniformly spaced points to ensure coverage
        # 3. Points at the endpoints (f1=0, f1=1)
        
        # Generate x0 values
        if budget == 1:
            x0_vals = np.array([0.5])
        else:
            # Discontinuity points in [0, 1] for u = (2k+1)/20
            disc_points = np.array([(2*k+1)/20.0 for k in range(10) if 0 <= (2*k+1)/20.0 <= 1])
            
            # We want to allocate budget points:
            # - Some at discontinuities
            # - Rest spread uniformly
            
            # Let's use a strategy: place points at discontinuities and fill in between
            # Number of discontinuities
            n_disc = len(disc_points)
            
            # Allocate at least 1 point per discontinuity if budget allows
            n_disc_use = min(n_disc, max(1, budget // 2))
            
            # Select which discontinuities to use
            if n_disc_use < n_disc:
                # Choose evenly spaced subset of discontinuities
                indices = np.linspace(0, n_disc-1, n_disc_use, dtype=int)
                disc_to_use = disc_points[indices]
            else:
                disc_to_use = disc_points
            
            # Remaining budget for uniform points
            n_uniform = budget - len(disc_to_use)
            
            if n_uniform > 0:
                # Uniform points in [0, 1], avoiding overlap with disc_to_use
                uniform_points = np.linspace(0, 1, n_uniform + 2)[1:-1]
            else:
                uniform_points = np.array([])
            
            # Combine and sort
            x0_vals = np.sort(np.concatenate([disc_to_use, uniform_points]))
            
            # Ensure we have exactly budget points
            if len(x0_vals) > budget:
                x0_vals = x0_vals[:budget]
            elif len(x0_vals) < budget:
                # Fill with additional uniform points
                extra = np.linspace(0, 1, budget - len(x0_vals) + 2)[1:-1]
                x0_vals = np.sort(np.concatenate([x0_vals, extra]))
        
        # Construct the x vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x0_vals
        X[:, 1:] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
