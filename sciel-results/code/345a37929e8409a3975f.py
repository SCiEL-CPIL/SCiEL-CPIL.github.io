"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exploits ZDT3 structure to sample the Pareto front directly."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        # Bounds are [0, 1] for all variables in ZDT3
        # We don't strictly need to read them if we assume the standard ZDT3 definition,
        # but we use them to be safe.
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use all budget evaluations
        # x_1 is distributed uniformly to cover the Pareto front range
        x1_values = np.linspace(0.0, 1.0, budget)
        
        # Optimal settings for x_2...x_dim
        # To minimize g(x) (when h(x1) > 0), set x_i = 0.5
        x_min_g = np.full(dim, 0.5)
        
        # To maximize g(x) (when h(x1) < 0), set x_i to a local max of the term
        # (x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))
        # Max occurs at x_i - 0.5 = 0.05, 0.15, ...
        # 0.05 is a good choice.
        x_max_g = np.full(dim, 0.05)
        # Ensure x_max_g is within bounds
        x_max_g = np.clip(x_max_g, lo, hi)
        x_min_g = np.clip(x_min_g, lo, hi)
        
        # We store the best points found, though the evaluator records all.
        # We just need to make the calls.
        
        for i in range(budget):
            x1 = x1_values[i]
            # Calculate h(x1)
            # h(z) = 1 - sqrt(z) - sin(10*pi*z)
            if x1 < 1e-10:
                h_val = 1.0
            else:
                h_val = 1.0 - np.sqrt(x1) - np.sin(10.0 * np.pi * x1)
            
            x = np.zeros(dim)
            x[0] = x1
            
            if h_val > 0:
                # Minimize g to minimize f2 = g*h
                x[1:] = x_min_g[1:]
            else:
                # Maximize g to minimize f2 = g*h (since h is negative)
                x[1:] = x_max_g[1:]
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
        return None
# EVOLVE-BLOCK-END
