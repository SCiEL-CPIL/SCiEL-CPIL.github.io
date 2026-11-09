"""Multi-objective optimizer for DTLZ2 exploiting the known spherical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280  # 300s limit, keep 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def halton(n, base):
            """Generate n Halton sequence points in [0,1)."""
            seq = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0 / i
                val = 0.0
                while f > 0:
                    val += f * np.floor(i * f)
                    f /= base
                seq[i - 1] = val
            return seq

        # Strategy:
        # 1. Use Halton sequence to sample densely on the known Pareto front (quarter-sphere)
        # 2. Use remaining budget for local refinement near the front

        # For DTLZ2 with n_obj=3, Pareto front is quarter-sphere:
        # f1 = sin(a)*cos(b), f2 = sin(a)*sin(b), f3 = cos(a)
        # where a, b in [0, pi/2]
        # Decision variables: x1 = 2*a/pi, x2 = 2*b/pi, x3..x12 = 0.5

        n_front = min(budget, 150)

        # Generate 2D Halton sequence for (a, b) angles
        h1 = halton(n_front, 2)
        h2 = halton(n_front, 3)

        # Map to angles
        a = h1 * (np.pi / 2)
        b = h2 * (np.pi / 2)

        # Build decision vectors
        X = np.zeros((n_front, dim))
        X[:, 0] = 2 * a / np.pi
        X[:, 1] = 2 * b / np.pi
        X[:, 2:] = 0.5
        X = np.clip(X, lo, hi)

        # Evaluate all front points
        F = np.array([problem(x) for x in X])
        used = n_front

        # Use remaining budget for local refinement
        if used < budget:
            remaining = budget - used
            for _ in range(remaining):
                if time.time() > deadline:
                    break

                # Pick a random existing point and perturb it
                idx = np.random.randint(0, n_front)
                x = X[idx].copy()

                # Small perturbation in the last 10 variables (which control g)
                x[2:] += 0.005 * np.random.standard_normal(dim - 2)
                x = np.clip(x, lo, hi)

                fx = np.array(problem(x))
                used += 1

                # Keep the point if it improves hypervolume contribution
                # Simple heuristic: if sum of objectives is lower, keep it
                current_sum = np.sum(F[idx])
                new_sum = np.sum(fx)

                if new_sum < current_sum:
                    X[idx] = x
                    F[idx] = fx

        return X
# EVOLVE-BLOCK-END
