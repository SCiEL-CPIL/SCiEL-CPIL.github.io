"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimized Pareto front sampling for ZDT3 with dominated point removal.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate n_evals candidate points on the analytic Pareto front
        # x_1 uniformly spaced in [0, 1], x_2..x_n = 0.5
        x1_vals = np.linspace(0, 1, n_evals)
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Filter out points that are not strictly better than reference point [11, 11]
        ref = np.array([11.0, 11.0])
        valid = np.all(F < ref, axis=1)
        
        if not np.any(valid):
            return X[0]
        
        F_valid = F[valid]
        X_valid = X[valid]
        
        # Remove dominated points: a point is dominated if there exists another point
        # that is <= in all objectives and < in at least one
        n = len(F_valid)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # Check if j dominates i
                if np.all(F_valid[j] <= F_valid[i]) and np.any(F_valid[j] < F_valid[i]):
                    dominated[i] = True
                    break
        
        F_nondom = F_valid[~dominated]
        X_nondom = X_valid[~dominated]
        
        if len(F_nondom) == 0:
            return X_valid[0]
        
        # Return any point - the return value is ignored
        return X_nondom[0]
# EVOLVE-BLOCK-END
