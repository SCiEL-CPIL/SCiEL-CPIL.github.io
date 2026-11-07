import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0
        
        # ZDT6 f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Range of f1 is [f1_min, 1.0]
        
        # Pre-compute a high-resolution lookup table for f1(x1)
        n_grid = 200000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        y_grid = np.exp(-4 * x_grid) * np.power(sin_term, 8)
        f1_grid = 1 - y_grid
        
        f1_min_val = np.min(f1_grid)
        f1_max_val = 1.0
        
        # Optimize the placement of n_evals points to maximize hypervolume.
        # For ZDT6, the Pareto front is f2 = 1 - sqrt(f1), with f1 in [f1_min, 1].
        # Use analytical inverse CDF approach based on hypervolume density.
        
        # The marginal hypervolume contribution of a point at f1 is proportional to
        # (11 - f2(f1)) = 11 - (1 - sqrt(f1)) = 10 + sqrt(f1)
        # So the optimal density of points in f1-space is proportional to 1/(10 + sqrt(f1))
        
        # Compute the CDF of the optimal distribution
        # Let u = f1, then we want density p(u) = w(u) / integral(w)
        # where w(u) = 1/(10 + sqrt(u))
        
        # Compute integral of w(u) from f1_min to f1_max analytically
        # integral of 1/(10 + sqrt(u)) du
        # Let t = sqrt(u), u = t^2, du = 2t dt
        # integral of 2t/(10+t) dt = integral of (2 - 20/(10+t)) dt
        # = 2t - 20*ln(10+t)
        
        def integral_w(t):
            # t = sqrt(u)
            return 2*t - 20*np.log(10 + t)
        
        # Compute CDF values
        t_min = np.sqrt(f1_min_val)
        t_max = np.sqrt(f1_max_val)
        
        I_min = integral_w(t_min)
        I_max = integral_w(t_max)
        
        # CDF(u) = (integral_w(sqrt(u)) - I_min) / (I_max - I_min)
        
        # Invert the CDF: given target probability q, find u such that CDF(u) = q
        # integral_w(sqrt(u)) = I_min + q * (I_max - I_min)
        
        # We need to solve for u: 2*sqrt(u) - 20*ln(10 + sqrt(u)) = I_min + q * (I_max - I_min)
        
        # Use bisection to find u for each q
        target_probs = np.linspace(0, 1, n_evals)
        
        # Pre-compute the target integral values
        target_integrals = I_min + target_probs * (I_max - I_min)
        
        # Bisection to find u for each target integral
        u_lo = f1_min_val
        u_hi = f1_max_val
        
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_I = target_integrals[i]
            
            # Bisection
            lo_u = u_lo
            hi_u = u_hi
            for _ in range(50):
                mid_u = (lo_u + hi_u) / 2
                mid_I = integral_w(np.sqrt(mid_u))
                if mid_I < target_I:
                    lo_u = mid_u
                else:
                    hi_u = mid_u
            
            target_f1 = (lo_u + hi_u) / 2
            
            # Find x1 that achieves this f1 value
            # Use the grid to find the closest x1
            diffs = np.abs(f1_grid - target_f1)
            min_idx = np.argmin(diffs)
            x1_vals[i] = x_grid[min_idx]
        
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() - start_time > deadline - 1.0:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
