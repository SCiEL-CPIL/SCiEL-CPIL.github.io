"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within a shrinking hyper-sphere for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # WFG9 has known Pareto front near the origin in decision space
        # The bounds for WFG9 are typically [-50, 50] per dimension
        # We sample within a hyper-sphere centered at origin with shrinking radius

        # Initial radius should cover the bounds
        max_radius = np.max(hi - lo) / 2.0
        # But for WFG9, the interesting region is much smaller
        # Use a radius that starts moderate and shrinks
        initial_radius = 10.0  # WFG9 Pareto front is near origin
        final_radius = 0.1

        # We'll use a linear shrinkage over the budget
        # But we also need to stay within bounds

        # Generate points on the surface of a hyper-sphere (or in the ball)
        # For better coverage, sample in the ball (volume) not just surface

        rng = np.random.default_rng(0)

        # We'll do random sampling within the shrinking ball
        # Each sample: generate a random direction, then a random radius scaled by current radius

        # To sample uniformly in a ball:
        # 1. Generate random direction (normalized Gaussian)
        # 2. Generate radius r = R * (U)^(1/dim) where U ~ Uniform(0,1)

        X = []
        F = []

        # We'll use all budget evaluations for sampling
        # But we need to be careful about time

        for i in range(budget):
            if time.time() > deadline:
                break

            # Linear shrinkage: radius goes from initial_radius to final_radius
            t = i / max(1, budget - 1)
            current_radius = initial_radius * (1 - t) + final_radius * t

            # Generate random direction
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = rng.standard_normal(dim)
                norm = np.linalg.norm(direction)
            direction = direction / norm

            # Sample radius uniformly in the ball
            u = rng.random()
            r = current_radius * (u ** (1.0 / dim))

            x = r * direction

            # Clip to bounds
            x = np.clip(x, lo, hi)

            # Evaluate
            f = np.array(problem(x))
            X.append(x)
            F.append(f)

        if len(X) == 0:
            # Fallback: return a single point
            x = np.clip(np.zeros(dim), lo, hi)
            f = np.array(problem(x))
            X = [x]
            F = [f]

        X = np.array(X)
        F = np.array(F)

        return X
# EVOLVE-BLOCK-END
