"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within a shrinking hypersphere for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        # Scale factor: WFG9 typically has bounds like [0,1] or similar
        # The Pareto front is near the origin in the normalized space
        # We'll work in the normalized [0,1]^d space and map back

        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Use numpy random (seeded by evaluator)
        # But we also need our own rng for internal operations
        # Actually, the evaluator seeds np.random, so we can use np.random directly
        # But let's use a local rng for reproducibility within our call
        rng = np.random.default_rng()

        # We'll sample in the unit hypercube [0,1]^dim
        # The WFG9 Pareto front is located near the origin in the transformed space
        # Actually, for WFG9, the Pareto front is defined in terms of the objectives
        # Let's just sample within a shrinking hypersphere centered at the midpoint

        # Strategy: sample points uniformly within a hypersphere of radius r(t)
        # r(t) shrinks linearly from r_max to r_min over the budget
        r_max = np.sqrt(dim) * 0.5  # radius that covers most of the hypercube
        r_min = 0.1  # small radius near the center

        # Store all evaluated points and their objectives
        X_list = []
        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            # x is in [0,1]^dim, map to actual bounds
            x_actual = lo + (hi - lo) * x
            f = problem(x_actual)
            used += 1
            return f

        # Initial population: sample within the full hypersphere
        pop_size = min(50, budget // 4)
        for i in range(pop_size):
            # Sample uniformly in hypersphere of radius r_max
            # Generate random direction and radius
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            # Cube root for uniform volume distribution
            r = r_max * rng.random() ** (1.0 / dim)
            x = center + r * half_range * direction
            # Clip to bounds
            x = np.clip(x, lo, hi)
            # Normalize to [0,1] for storage
            x_norm = (x - lo) / (hi - lo)
            f = evaluate(x)
            X_list.append(x_norm)
            F_list.append(f)

        # Main loop: sample within shrinking hypersphere
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute current radius (linear shrink)
            progress = used / max(budget, 1)
            r_current = r_max + (r_min - r_max) * progress

            # Sample new point within hypersphere of radius r_current
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.ones(dim) / np.sqrt(dim)
            r = r_current * rng.random() ** (1.0 / dim)
            x = center + r * half_range * direction
            x = np.clip(x, lo, hi)
            x_norm = (x - lo) / (hi - lo)

            f = evaluate(x)
            X_list.append(x_norm)
            F_list.append(f)

        # Return the last evaluated point (return value is ignored by evaluator)
        if X_list:
            return X_list[-1]
        return np.zeros(dim)
# EVOLVE-BLOCK-END
