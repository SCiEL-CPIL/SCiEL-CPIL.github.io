"""Multi-objective optimizer on WFG9 using constrained random search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search with shrinking hypersphere for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.RandomState(42)

        # WFG9 Pareto front is near the origin in decision space
        # Use shrinking radius centered at origin (or midpoint if bounds don't include 0)
        center = np.zeros(dim)
        for d in range(dim):
            if lo[d] > 0:
                center[d] = lo[d]
            elif hi[d] < 0:
                center[d] = hi[d]
        
        # Initial radius based on distance from center to bounds
        max_radius = 0.0
        for d in range(dim):
            r1 = abs(center[d] - lo[d])
            r2 = abs(center[d] - hi[d])
            max_radius = max(max_radius, r1, r2)
        
        if max_radius < 1e-10:
            max_radius = 1.0

        # Total budget split: use all 200 evaluations
        # Radius shrinks geometrically
        n_evals = budget
        # Start with full radius, shrink over time
        # Use a schedule where radius = max_radius * (1 - t)^alpha
        # where t goes from 0 to 1 over all evaluations

        X_best = []
        F_best = []

        for i in range(n_evals):
            if time.time() > deadline:
                break

            t = i / max(1, n_evals - 1)
            # Geometric shrinkage: radius decreases from max_radius to small value
            # Use exponential decay for faster convergence
            radius = max_radius * np.exp(-3.0 * t)
            # Ensure minimum radius
            radius = max(radius, 1e-6)

            # Sample point in hypersphere
            # Generate random direction
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.ones(dim) / dim
            else:
                direction = direction / norm

            # Random radius within [0, radius]
            r = radius * rng.random() ** (1.0 / dim)  # Uniform in hypersphere volume
            point = center + r * direction

            # Clip to bounds
            point = np.clip(point, lo, hi)

            f = problem(point)
            F_best.append(f)
            X_best.append(point)

        return np.array(X_best) if X_best else np.array([])
# EVOLVE-BLOCK-END
