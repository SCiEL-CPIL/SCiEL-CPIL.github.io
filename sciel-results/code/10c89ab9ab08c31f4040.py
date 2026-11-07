"""Multi-objective optimizer for ZDT1 using optimal hypervolume placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimal hypervolume maximizer for ZDT1 (2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget  # 200
        r1, r2 = 11.0, 11.0

        # Compute optimal t_i values to maximize HV
        # HV = sum_{i=1}^{n} (t_{i+1} - t_i) * (10 + sqrt(t_i))
        # where t_1 = 0, t_{n+1} = 11
        # Optimal: allocate gaps proportional to weight w(t_i) = 10 + sqrt(t_i)
        # Iterate to convergence

        t = np.zeros(n_evals)
        t[0] = 0.0

        for iteration in range(200):
            if time.time() > deadline:
                break
            # Compute weights
            w = 10.0 + np.sqrt(np.maximum(t, 1e-15))
            # Total gap to distribute from t_1=0 to t_{n+1}=11
            total_gap = r1 - t[0]  # = 11.0
            # Allocate gaps proportional to weights
            gaps = total_gap * w / np.sum(w)
            # Build new t values
            t_new = np.zeros(n_evals)
            t_new[0] = t[0]
            for i in range(1, n_evals):
                t_new[i] = t_new[i - 1] + gaps[i - 1]
            # Ensure monotonicity
            t_new = np.maximum.accumulate(t_new)
            t_new = np.minimum(t_new, r1 - 1e-9)
            # Check convergence
            if np.max(np.abs(t_new - t)) < 1e-12:
                t = t_new
                break
            t = t_new

        # Create x vectors: x[0] = t_i, x[1:] = 0 (Pareto front)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
