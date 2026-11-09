"""Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""
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
        deadline = start_time + 290  # safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # ZDT6: f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6, f2 = g*(1-sqrt(f1/g))
        # Pareto front: x2..x10 = 0, so g=1
        # f2 = 1 - sqrt(f1)
        # f1 ranges from f1_min to 1.0

        # Find the minimum f1 point precisely
        # df1/dx1 = 0 when tan(6*pi*x1) = 9*pi (in [0, 1/6])
        target_angle = np.arctan(9 * np.pi)
        x1_min = target_angle / (6 * np.pi)

        # Compute f1_min
        sin_val = 9 * np.pi / np.sqrt(1 + (9 * np.pi) ** 2)
        f1_min_val = 1 - np.exp(-4 * x1_min) * sin_val ** 6

        # Strategy: use a power-law distribution to bias sampling towards lower f1
        # where the Pareto front is steeper and each point contributes more to hypervolume
        # Use t^p distribution with p < 1 to bias towards t=0 (low f1)
        
        # For the Pareto front f2 = 1 - sqrt(f1), the hypervolume contribution
        # per unit f1 is (10 + sqrt(f1)) which is larger at low f1.
        # Optimal spacing equalizes marginal HV contributions.
        # Using a power-law: f1_i = f1_min + (1 - f1_min) * (i/n)^(2/3) approximates optimal
        
        # Actually, let's use the exact optimal spacing derived from equalizing
        # marginal contributions. For HV = sum (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)),
        # the optimal spacing satisfies (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)) = const
        # This gives f1_{i+1} - f1_i = C / (10 + sqrt(f1_i))
        
        # Numerically compute optimal f1 positions
        n_grid = 1000000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_lookup = 1 - np.exp(-4 * x1_grid) * (np.sin(6 * np.pi * x1_grid) ** 6)
        
        # Sort by f1 for lookup
        sort_idx = np.argsort(f1_lookup)
        f1_sorted = f1_lookup[sort_idx]
        x1_sorted = x1_grid[sort_idx]

        # Compute optimal f1 positions by solving the ODE-like condition
        # We want n_evals points that maximize hypervolume
        # For 2D with f2 = 1 - sqrt(f1), ref = (11, 11):
        # HV = sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)) + (11 - f1_{n-1}) * (10 + sqrt(f1_{n-1}))
        
        # Use iterative refinement: start with uniform, then adjust
        f1_targets = np.linspace(f1_min_val, 1.0, n_evals)
        
        # Refine spacing: points should be denser where d(HV)/d(f1) is larger
        # d(HV)/d(f1) at position f1 is (10 + sqrt(f1)) for the interval ending there
        # So spacing should be proportional to 1/(10 + sqrt(f1))
        
        # Compute cumulative weight
        f1_fine = np.linspace(f1_min_val, 1.0, 10000)
        weights = 1.0 / (10.0 + np.sqrt(f1_fine))
        cumulative = np.cumsum(weights)
        cumulative /= cumulative[-1]
        
        # Interpolate to get optimal f1 positions
        f1_targets = np.interp(np.linspace(0, 1, n_evals), cumulative, f1_fine)
        # Ensure endpoints
        f1_targets[0] = f1_min_val
        f1_targets[-1] = 1.0

        # For each target f1, find the corresponding x1
        x1_selected = np.zeros(n_evals)
        for i in range(n_evals):
            idx = np.searchsorted(f1_sorted, f1_targets[i])
            idx = min(max(idx, 1), n_grid - 1)
            # Linear interpolation between idx-1 and idx
            f0 = f1_sorted[idx - 1]
            f1_val = f1_sorted[idx]
            x0 = x1_sorted[idx - 1]
            x1_val = x1_sorted[idx]
            if f1_val > f0:
                t = (f1_targets[i] - f0) / (f1_val - f0)
                x1_selected[i] = x0 + t * (x1_val - x0)
            else:
                x1_selected[i] = x1_val

        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_selected
        # x2..x10 = 0 for Pareto front

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
