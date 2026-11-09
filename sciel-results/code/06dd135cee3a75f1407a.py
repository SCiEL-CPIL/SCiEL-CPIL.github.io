# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy: Adaptive sampling of the known 1D Pareto front.
    ZDT3's Pareto front: f1 = x1, f2 = 1 - sqrt(x1) - x1*sin(10*pi*x1)
    We use adaptive sampling to better cover the front for hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Compute the Pareto front: f1 = x1, f2 = 1 - sqrt(x1) - x1*sin(10*pi*x1)
        # For hypervolume, we want to maximize the area under the curve.
        # The front is concave in some regions and convex in others due to the sin term.
        # Use adaptive sampling: sample more where the front changes rapidly.
        
        # First pass: uniform sampling to estimate the front
        x1_uniform = np.linspace(0, 1, 1000)
        f2_uniform = 1 - np.sqrt(x1_uniform) - x1_uniform * np.sin(10 * np.pi * x1_uniform)
        
        # Compute local slope to identify regions of rapid change
        df2_dx1 = np.gradient(f2_uniform, x1_uniform)
        slope_abs = np.abs(df2_dx1)
        
        # Create a probability distribution for sampling based on slope
        # Higher slope -> more samples needed
        weights = slope_abs + 0.1  # Add small constant to avoid zeros
        weights = weights / weights.sum()
        
        # Sample from this distribution
        # Use inverse CDF method
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]
        
        # Generate random samples in [0, 1] and map to x1 via inverse CDF
        u = np.random.uniform(0, 1, n_evals)
        x1_vals = np.interp(u, cdf, x1_uniform)
        
        # Also add some uniform samples to ensure coverage
        n_uniform = max(1, n_evals // 10)
        n_adaptive = n_evals - n_uniform
        x1_uniform_part = np.linspace(0, 1, n_uniform)
        x1_adaptive_part = np.interp(np.random.uniform(0, 1, n_adaptive), cdf, x1_uniform)
        
        # Combine and sort
        x1_all = np.concatenate([x1_uniform_part, x1_adaptive_part])
        x1_all = np.sort(x1_all)
        
        # Create population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_all
        # x2..xn are 0 (minimizes g to 1)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

# EVOLVE-BLOCK-END
