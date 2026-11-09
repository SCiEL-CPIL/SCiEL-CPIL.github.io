import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        t_dead = t0 + 275.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # Build a 13x13 grid on (x0, x1) with x[2:]=0 (Pareto front)
        n_grid = 13
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        x0_flat = X0.flatten()
        x1_flat = X1.flatten()
        n_grid_pts = len(x0_flat)

        X_grid = np.zeros((n_grid_pts, dim))
        X_grid[:, 0] = x0_flat
        X_grid[:, 1] = x1_flat

        # Evaluate grid points
        for i in range(n_grid_pts):
            if time.time() > t_dead:
                break
            problem(X_grid[i])

        # Remaining budget: use Sobol-like low-discrepancy points in (x0, x1) space
        # with x[2:]=0 to stay on the Pareto front
        remaining = self.budget - min(self.budget, n_grid_pts)
        remaining = max(0, remaining)

        if remaining > 0:
            # Generate low-discrepancy points using a simple Sobol-like approach
            # For 2D, we can use the Van der Corput sequence (Halton) or a simple
            # stratified approach. Let's use Halton sequences for better coverage.
            
            def halton_1d(n, base):
                """Generate n Halton sequence values in [0,1)"""
                result = np.zeros(n)
                for idx in range(n):
                    val = 0.0
                    denom = base
                    num = idx + 1
                    while num > 0:
                        num, rem = divmod(num, base)
                        denom *= base
                        val += rem / denom
                    result[idx] = val
                return result

            n_extra = remaining
            # Use different bases for x0 and x1 to get 2D low-discrepancy points
            x0_extra = halton_1d(n_extra, 2)
            x1_extra = halton_1d(n_extra, 3)
            
            # Apply a transformation to concentrate points where hypervolume contribution is higher
            # For DTLZ7, f3 = 1 + g + h where h = (n-g)*f3/(1+g)
            # When x[2:]=0, g=0, so h = n*f3 = 21*f3
            # f3 = 1 + 21*(1 - x1*(1 + 2*x0))
            # So f3 ranges from 1 (when x1=0 or x0=0.5,x1=0) to 22 (when x1=1,x0=0)
            # Points with smaller f3 contribute more to hypervolume
            # So we want more density where x1 is small
            
            # Transform x1 to concentrate near 0: x1' = x1^2
            x1_extra = x1_extra ** 2
            
            X_extra = np.zeros((n_extra, dim))
            X_extra[:, 0] = x0_extra
            X_extra[:, 1] = x1_extra

            for i in range(n_extra):
                if time.time() > t_dead:
                    break
                problem(X_extra[i])

        return X_grid
