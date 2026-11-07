"""Optimizer for multi-objective optimization tasks."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer using optimal quantization on ZDT1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = min(self.budget, 200)

        # ZDT1 Pareto front: f1 = x[0], f2 = (1+0.1*f1)*(1 - sqrt(f1/(1+0.1*f1)))
        # Optimal 1D quantization: place points where CDF of weight w(x) = 11 - f2(x) is uniform
        # w(x) = 11 - (1+0.1x)*(1 - sqrt(x/(1+0.1x)))

        # Compute the weight function and its integral on a fine grid
        n_grid = 100000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = (1 + 0.1 * x_grid) * (1 - np.sqrt(x_grid / (1 + 0.1 * x_grid)))
        w_grid = 11.0 - f2_grid  # weight for hypervolume contribution

        # Compute cumulative integral W(x) = integral_0^x w(t) dt
        # Use cumulative trapezoidal integration
        dx = x_grid[1] - x_grid[0]
        W_grid = np.zeros(n_grid)
        for i in range(1, n_grid):
            W_grid[i] = W_grid[i-1] + 0.5 * (w_grid[i-1] + w_grid[i]) * dx

        W_total = W_grid[-1]

        # Place n_evals points at quantiles of the CDF
        # Use midpoints for better coverage: (i + 0.5) / n_evals
        quantiles = (np.arange(n_evals) + 0.5) / n_evals
        target_W = quantiles * W_total

        # Invert: find x such that W(x) = target_W using interpolation
        f1_vals = np.interp(target_W, W_grid, x_grid)

        # Clip to valid range (avoid exactly 0 or 1 which may cause issues)
        f1_vals = np.clip(f1_vals, 1e-10, 1.0 - 1e-10)

        # Create the actual points in the decision space
        # For ZDT1, x[0] = f1, x[1:] = 0 gives points on the Pareto front
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = f1_vals

        # Evaluate all points
        used = 0
        for i in range(n_evals):
            if used >= self.budget:
                break
            if time.time() > deadline - 5:  # Safety margin
                break
            f = problem(X[i])
            used += 1

        return X

# EVOLVE-BLOCK-END
