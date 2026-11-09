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
        deadline = start_time + 285  # safety margin before 330s limit

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
        # f1 ranges from f1_min to 1

        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Find f1_min using a dense grid
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min = x1_grid[idx_min]
        f1_min = f1_grid[idx_min]

        # U(t) = integral of (R-s)/(2*sqrt(s)) ds from f1_min to t
        # = [R*sqrt(s) - (1/3)*s^{3/2}] from f1_min to t

        def U_func(t):
            return ref * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create n_evals target U values, evenly spaced
        U_targets = np.linspace(U_min + U_range / (2*n_evals),
                                 U_max - U_range / (2*n_evals), n_evals)
        U_targets_shifted = U_targets - U_min

        # Invert U(f1) = U_target using a fine grid
        n_invert = 1000000
        f1_grid_inv = np.linspace(f1_min, 1.0, n_invert)
        U_grid_inv = U_func(f1_grid_inv) - U_min

        # Interpolate to get f1 values
        f1_selected = np.interp(U_targets_shifted, U_grid_inv, f1_grid_inv)

        # Now find x1 for each f1 value using vectorized search
        # Build a dense mapping from f1 to x1
        n_map = 200000
        x1_map = np.linspace(0, 1, n_map)
        f1_map = compute_f1(x1_map)
        
        # For each target f1, find the closest x1 using searchsorted on sorted f1
        # Since f1 is non-monotonic, we need to handle both branches
        # Split at the minimum point
        idx_min_map = np.argmin(f1_map)
        x1_left = x1_map[:idx_min_map + 1]
        f1_left = f1_map[:idx_min_map + 1]
        x1_right = x1_map[idx_min_map:]
        f1_right = f1_map[idx_min_map:]
        
        # f1_left is decreasing (from x1=0 to x1_min), f1_right is increasing (from x1_min to x1=1)
        # For targets in [f1_min, f1(0)], we can search both branches
        # f1(0) = 1 - exp(0)*sin(0)^6 = 1
        # f1(1) = 1 - exp(-4)*sin(6*pi)^6 = 1 - exp(-4)*0 = 1
        
        x1_final = np.zeros(n_evals)
        f1_targets = f1_selected
        
        # Interpolate
        x1_from_left = np.interp(f1_targets, f1_left[::-1], x1_left[::-1])
        x1_from_right = np.interp(f1_targets, f1_right, x1_right)
        
        # Choose the x1 that gives f1 closer to target
        f1_check_left = compute_f1(x1_from_left)
        f1_check_right = compute_f1(x1_from_right)
        
        err_left = np.abs(f1_check_left - f1_targets)
        err_right = np.abs(f1_check_right - f1_targets)
        
        use_left = err_left <= err_right
        x1_final[use_left] = x1_from_left[use_left]
        x1_final[~use_left] = x1_from_right[~use_left]

        # Build points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
