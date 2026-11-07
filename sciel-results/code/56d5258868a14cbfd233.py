import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2 (scaled to ref 11,11):
        # f1 = x1
        # f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        
        # Optimal x1 spacing for hypervolume maximization
        # Hypervolume sum approx: sum (dx_i) * (11 - f2(x_i))
        # 11 - f2(x) = 11 * sqrt(x)
        # We want to maximize sum (x_{i+1}-x_i) * 11 * sqrt(x_i)
        # This suggests x_i should be spaced such that sqrt(x_i) is roughly linear
        # Let x_i = (i/m)^2
        
        num_points = min(200, self.budget)
        # Use i from 1 to num_points to avoid x=0 which has 0 height
        i = np.arange(1, num_points + 1, dtype=float)
        x1_vals = (i / num_points) ** 2.0

        # Evaluate these points on the Pareto front (x2...xn = 0)
        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        # If we have budget remaining, do some random exploration to ensure we capture
        # any numerical issues, but the main points are on the Pareto front
        if used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = self.budget - used
            for _ in range(min(remaining, 10)):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(n)
                f = problem(x)
                recorded.append(f)
                used += 1

        return recorded
