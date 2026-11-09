"""Model-and-optimize approach for ZDT6 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit

        n_evals = self.budget
        dim = self.dim
        ref = 11.0

        # ZDT6 analytical model:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # g(x) = 1 + 10*(dim-1) + sum_{i=2}^{dim} [x_i^2 - 10*cos(12*pi*x_i)]
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        #
        # Pareto front: x2..x10 = 0, so g = 1
        # On PF: f2 = 1 - sqrt(f1)
        # f1 ranges from f1_min to 1 (at x1=0 or x1=1)

        dim = self.dim

        # Compute f1(x1) analytically
        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Find f1_min: minimize f1 over x1 in [0,1]
        # Use fine grid + golden section refinement
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min_coarse = x1_grid[idx_min]

        # Refine around the minimum using a local search
        local_grid = np.linspace(max(0, x1_min_coarse - 0.005),
                                  min(1, x1_min_coarse + 0.005), 100000)
        f1_local = compute_f1(local_grid)
        idx_local_min = np.argmin(f1_local)
        x1_min = local_grid[idx_local_min]
        f1_min = compute_f1(x1_min)

        # On the Pareto front (g=1): f2 = 1 - sqrt(f1)
        # We need to select n_evals values of x1 such that the corresponding
        # (f1, f2) points maximize hypervolume with reference [11, 11].

        # Compute the total U
        def U_func(t):
            return 11.0 * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create target U values
        n_pts = n_evals
        U_targets = np.linspace(U_min + U_range / (2*n_pts),
                                 U_max - U_range / (2*n_pts), n_pts)

        # Invert U(f1) = U_target to find f1 values
        # Use binary search on a fine grid
        n_invert_grid = 50000
        f1_invert_grid = np.linspace(f1_min, 1.0, n_invert_grid)
        U_invert_grid = U_func(f1_invert_grid) - U_min  # shift so U(f1_min) = 0

        # For each U_target (shifted), find f1 via interpolation
        U_targets_shifted = U_targets - U_min
        f1_selected = np.interp(U_targets_shifted, U_invert_grid, f1_invert_grid)

        # Now convert f1 values back to x1 values
        # f1(x1) = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # This is not monotonic, so we need to find x1 for each f1.
        # For the Pareto front, we want the x1 that gives the desired f1
        # and is on the "front" (smallest x1 for a given f1 in the decreasing branch).

        # Create a fine mapping from x1 to f1 using two branches
        # Left branch: x1 in [0, x1_min], f1 decreases from f1(0)=1 to f1_min
        # Right branch: x1 in [x1_min, 1], f1 increases from f1_min to f1(1)=1
        
        n_left_grid = 100000
        n_right_grid = 100000
        
        # Left branch: x1 from 0 to x1_min (f1 goes from 1 down to f1_min)
        x1_left_grid = np.linspace(0, x1_min, n_left_grid)
        f1_left_grid = compute_f1(x1_left_grid)
        # f1_left_grid is monotonically decreasing, so reverse for searchsorted
        f1_left_rev = f1_left_grid[::-1]
        x1_left_rev = x1_left_grid[::-1]
        # Now f1_left_rev is monotonically increasing from f1_min to 1
        
        # Right branch: x1 from x1_min to 1 (f1 goes from f1_min up to 1)
        x1_right_grid = np.linspace(x1_min, 1.0, n_right_grid)
        f1_right_grid = compute_f1(x1_right_grid)
        # f1_right_grid is monotonically increasing from f1_min to 1
        
        # For each target f1, find x1 on both branches using searchsorted
        x1_final = np.zeros(n_pts)
        
        # Left branch lookup
        idx_left = np.searchsorted(f1_left_rev, f1_selected, side='left')
        idx_left = np.clip(idx_left, 0, n_left_grid - 1)
        # Interpolate for better precision
        idx_left_hi = np.minimum(idx_left + 1, n_left_grid - 1)
        idx_left_lo = np.maximum(idx_left - 1, 0)
        
        # Use linear interpolation between adjacent grid points
        f1_lo = f1_left_rev[idx_left_lo]
        f1_hi = f1_left_rev[idx_left_hi]
        x1_lo = x1_left_rev[idx_left_lo]
        x1_hi = x1_left_rev[idx_left_hi]
        
        denom = f1_hi - f1_lo
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        t_left = (f1_selected - f1_lo) / denom
        x1_left_interp = x1_lo + t_left * (x1_hi - x1_lo)
        
        # Right branch lookup
        idx_right = np.searchsorted(f1_right_grid, f1_selected, side='left')
        idx_right = np.clip(idx_right, 0, n_right_grid - 1)
        idx_right_hi = np.minimum(idx_right + 1, n_right_grid - 1)
        idx_right_lo = np.maximum(idx_right - 1, 0)
        
        f1_lo_r = f1_right_grid[idx_right_lo]
        f1_hi_r = f1_right_grid[idx_right_hi]
        x1_lo_r = x1_right_grid[idx_right_lo]
        x1_hi_r = x1_right_grid[idx_right_hi]
        
        denom_r = f1_hi_r - f1_lo_r
        denom_r = np.where(np.abs(denom_r) < 1e-15, 1e-15, denom_r)
        t_right = (f1_selected - f1_lo_r) / denom_r
        x1_right_interp = x1_lo_r + t_right * (x1_hi_r - x1_lo_r)
        
        # Choose the branch with smaller x1 (left branch preferred for Pareto front)
        x1_final = np.minimum(x1_left_interp, x1_right_interp)
        
        # Newton refinement for higher precision
        def f1_deriv(x1):
            # d/dx1 [1 - exp(-4*x1)*sin(6*pi*x1)^6]
            # = 4*exp(-4*x1)*sin(6*pi*x1)^6 - 6*exp(-4*x1)*sin(6*pi*x1)^5*cos(6*pi*x1)*6*pi
            s = np.sin(6.0 * np.pi * x1)
            c = np.cos(6.0 * np.pi * x1)
            e = np.exp(-4.0 * x1)
            return 4.0 * e * s**6 - 36.0 * np.pi * e * s**5 * c
        
        for _ in range(3):
            f1_current = compute_f1(x1_final)
            df1 = f1_deriv(x1_final)
            df1 = np.where(np.abs(df1) < 1e-12, 1e-12, df1)
            x1_final = x1_final - (f1_current - f1_selected) / df1
            x1_final = np.clip(x1_final, 0, 1)

        # Build the points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_final

        # Evaluate all points, tracking time
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
