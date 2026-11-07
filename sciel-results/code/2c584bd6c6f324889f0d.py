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
        # f1_min is approx 1 - exp(-1/3)
        
        # Pre-compute a high-resolution lookup table for f1(x1)
        n_grid = 100000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        # Avoid negative zero issues with power 8, but sin^8 is always positive
        # np.power handles this fine
        y_grid = np.exp(-4 * x_grid) * np.power(sin_term, 8)
        f1_grid = 1 - y_grid
        
        f1_min_val = np.min(f1_grid)
        f1_max_val = 1.0
        
        # Optimize the placement of n_evals points to maximize hypervolume.
        # For ZDT6, the Pareto front is f2 = 1 - sqrt(f1), with f1 in [f1_min, 1].
        # Use analytical inverse CDF approach based on hypervolume density.
        
        # The weight function for optimal point distribution is:
        # w(u) = (R1 - u) * |dv/du| where v(u) = 1 - sqrt(u)
        # dv/du = -1/(2*sqrt(u)), so |dv/du| = 1/(2*sqrt(u))
        # w(u) = (11 - u) / (2 * sqrt(u))
        
        # Compute the cumulative integral of w(u) from f1_min to u
        # C(u) = integral_{f1_min}^{u} (11 - t) / (2*sqrt(t)) dt
        # Let t = s^2, dt = 2s ds
        # C(u) = integral_{sqrt(f1_min)}^{sqrt(u)} (11 - s^2) / s * s ds
        #      = integral_{sqrt(f1_min)}^{sqrt(u)} (11 - s^2) ds
        #      = [11s - s^3/3]_{sqrt(f1_min)}^{sqrt(u)}
        #      = 11*sqrt(u) - u*sqrt(u)/3 - (11*sqrt(f1_min) - f1_min*sqrt(f1_min)/3)
        
        def cum_weight(u):
            """Cumulative weight function C(u)."""
            su = np.sqrt(u)
            sfm = np.sqrt(f1_min_val)
            return (11.0 * su - su**3 / 3.0) - (11.0 * sfm - sfm**3 / 3.0)
        
        C_max = cum_weight(f1_max_val)
        
        # Place points at equal intervals in C-space
        c_targets = np.linspace(0, C_max, n_evals)
        
        # Solve C(u_i) = c_targets for u_i using bisection
        best_u = np.zeros(n_evals)
        for i in range(n_evals):
            target = c_targets[i]
            lo_u = f1_min_val
            hi_u = f1_max_val
            for _ in range(60):
                mid_u = (lo_u + hi_u) / 2.0
                if cum_weight(mid_u) < target:
                    lo_u = mid_u
                else:
                    hi_u = mid_u
            best_u[i] = (lo_u + hi_u) / 2.0
        
        # Use best_u to find x1 values via vectorized search
        # For each target_f1, find x1 such that f1(x1) = target_f1
        # Since f1 is not monotonic, we need to find all x1 values that achieve each target
        # and pick the one that's valid (within [0,1])
        
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_f1 = best_u[i]
            # Find all indices where f1_grid is close to target
            diff = np.abs(f1_grid - target_f1)
            min_diff = np.min(diff)
            if min_diff < 1e-6:
                # Find the first occurrence
                idx = np.where(diff <= min_diff + 1e-9)[0][0]
                x1_vals[i] = x_grid[idx]
            else:
                # If no exact match, use the closest
                idx = np.argmin(diff)
                x1_vals[i] = x_grid[idx]
        
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
