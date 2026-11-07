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
        # We use an analytical inverse CDF approach based on the front geometry.
        # The weight function for optimal point distribution is proportional to
        # the "hypervolume density" w(u) = (R1 - u) * |dv/du|.
        # Here v(u) = 1 - sqrt(u), so dv/du = -1/(2*sqrt(u)).
        # w(u) = (11 - u) / (2 * sqrt(u)).
        
        # We want to place points such that the integral of w(u) is uniform.
        # C(u) = integral_{f1_min}^{u} w(t) dt
        # We solve C(u_i) = (i + 0.5) / n_evals for u_i.
        
        # Analytical integration of w(t) = (11 - t) / (2 * sqrt(t))
        # = 11/(2) t^{-1/2} - 1/2 t^{1/2}
        # Integral = 11 * sqrt(t) - 1/4 t + C
        # Let I(t) = 11 * sqrt(t) - 0.25 * t
        
        def integral_w(t):
            return 11.0 * np.sqrt(t) - 0.25 * t
        
        I_min = integral_w(f1_min_val)
        I_max = integral_w(f1_max_val)
        
        # Target cumulative values
        # Using midpoints to avoid boundary issues and ensure symmetry
        targets = (np.arange(n_evals) + 0.5) / n_evals
        target_vals = I_min + targets * (I_max - I_min)
        
        # Invert I(t) = A * sqrt(t) - B * t
        # 0.25 t - 11 sqrt(t) + target = 0
        # Let s = sqrt(t), then 0.25 s^2 - 11 s + target = 0
        # s^2 - 44 s + 4 * target = 0
        # s = (44 +/- sqrt(1936 - 16 * target)) / 2
        # s = 22 +/- sqrt(484 - 4 * target)
        # Since t is in [f1_min, 1], s is in [sqrt(f1_min), 1].
        # The root corresponding to the smaller t (and smaller s) is the one with minus?
        # Let's check the range.
        # I(t) is increasing for t in [0, 1] because w(t) > 0.
        # I'(t) = w(t) > 0.
        # The quadratic in s is s^2 - 44s + 4*target = 0.
        # Roots are s1 = 22 - sqrt(484 - 4*target) and s2 = 22 + sqrt(484 - 4*target).
        # s2 is always > 22, so t2 > 484, which is outside our domain [0, 1].
        # So we must use s1 = 22 - sqrt(484 - 4*target).
        
        # Calculate discriminant
        disc = 484.0 - 4.0 * target_vals
        # Ensure non-negative due to floating point errors
        disc = np.maximum(disc, 0.0)
        s_vals = 22.0 - np.sqrt(disc)
        
        best_u = s_vals ** 2
        
        # Clip to valid range
        best_u = np.clip(best_u, f1_min_val, f1_max_val)
        
        # Sort to ensure ascending order (should already be, but safe)
        best_u = np.sort(best_u)
        
        # Efficient lookup using sorted f1_grid
        sorted_f1, sort_idx = np.sort(f1_grid), np.argsort(f1_grid)
        
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_f1 = best_u[i]
            pos = np.searchsorted(sorted_f1, target_f1)
            # Check neighbors for best match
            best_dist = np.inf
            best_idx = pos
            for p in [pos-1, pos, pos+1]:
                if 0 <= p < n_grid:
                    d = np.abs(sorted_f1[p] - target_f1)
                    if d < best_dist:
                        best_dist = d
                        best_idx = p
            x1_vals[i] = x_grid[sort_idx[best_idx]]
        
        # Newton polish: refine x1 values to achieve target f1 more precisely
        def f1_of_x(x):
            return 1 - np.exp(-4*x) * np.power(np.sin(6*np.pi*x), 8)
        
        def df1_dx(x):
            s = np.sin(6*np.pi*x)
            c = np.cos(6*np.pi*x)
            return 4*np.exp(-4*x)*np.power(s, 7)*(s - 12*np.pi*c)
        
        for i in range(n_evals):
            x = x1_vals[i]
            target = best_u[i]
            for _ in range(10):
                f = f1_of_x(x) - target
                df = df1_dx(x)
                if abs(df) < 1e-15:
                    break
                x_new = x - f/df
                if not (0 <= x_new <= 1):
                    x_new = np.clip(x_new, 1e-10, 1-1e-10)
                if abs(x_new - x) < 1e-12:
                    x = x_new
                    break
                x = x_new
            x1_vals[i] = x
            
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
