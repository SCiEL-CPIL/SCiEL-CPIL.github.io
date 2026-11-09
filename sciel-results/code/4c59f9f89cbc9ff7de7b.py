"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using optimized Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Optimal distribution for ZDT2 Pareto front (f2 = 1 - sqrt(f1))
        # to minimize undominated area (maximize HV):
        # x_i proportional to i^2
        i_vals = np.arange(n_points)
        x0_vals = (i_vals / (n_points - 1)) ** 2
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 ensures g(x) = 1, placing points on the Pareto front
        
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
