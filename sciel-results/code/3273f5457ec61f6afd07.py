"""Multi-objective optimizer on WFG7 using a decomposition-based MOEA/D approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-style multi-objective optimizer with adaptive mutation and reference-point weighting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        N = min(60, max(20, self.budget // 5))
        N = int(N)

        # Create weight vectors using Tchebycheff-style approach
        # For 3 objectives, use simplex lattice
        weights = self._generate_weights(N, self.n_obj)

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Reference point for normalization (use max seen or upper bound)
        F_max = F.max(axis=0) + 1.0
        F_min = F.min(axis=0)

        # Track best hypervolume-contributing points
        best_points = list(F)
        best_xs = list(X)

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 10.0:
                break

            # For each weight vector, find the best individual in the neighborhood
            # and try to improve it
            for i in range(N):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                w = weights[i]

                # Find the individual that best matches this weight direction
                # Use normalized objectives
                F_norm = (F - F_min) / (F_max - F_min + 1e-10)
                # Tchebycheff aggregation: max_j w_j * |f_j - z_j|
                # Simplified: minimize weighted sum of normalized objectives
                scores = np.sum(w * F_norm, axis=1)
                parent_idx = int(np.argmin(scores))

                # Mutate
                # Adaptive mutation rate
                gen = used / max(N, 1)
                mutation_scale = 0.3 * (1.0 + 0.5 * np.sin(gen * 0.1))
                mutation = mutation_scale * span * rng.standard_normal(self.dim)
                # Add some crossover with another good individual
                other_idx = int(rng.integers(N))
                child = X[parent_idx] + 0.5 * (X[other_idx] - X[parent_idx]) + mutation
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update F_max and F_min
                F_max = np.maximum(F_max, fc)
                F_min = np.minimum(F_min, fc)

                # Replace the parent if the child is better in the weighted sense
                child_score = np.sum(w * ((fc - F_min) / (F_max - F_min + 1e-10)))
                parent_score = scores[parent_idx]
                if child_score <= parent_score + 1e-10:
                    X[parent_idx] = child
                    F[parent_idx] = fc
                    best_points.append(fc)
                    best_xs.append(child)

        return X

    def _generate_weights(self, N, n_obj):
        """Generate weight vectors on a simplex for n_obj objectives."""
        if n_obj == 3:
            # Generate points on the 2D simplex (triangle) for 3 objectives
            # Use a grid approach
            weights = []
            # Divide the simplex into a grid
            # Use a systematic sampling approach
            n_grid = int(np.ceil(np.sqrt(N)))
            for i in range(n_grid + 1):
                for j in range(n_grid + 1 - i):
                    k = n_grid - i - j
                    w = np.array([i / n_grid, j / n_grid, k / n_grid], dtype=float)
                    norm = np.sum(w)
                    if norm > 0:
                        w = w / norm
                    weights.append(w)
            # If we have too many, subsample
            if len(weights) > N:
                indices = np.random.default_rng(0).choice(len(weights), N, replace=False)
                weights = [weights[i] for i in indices]
            # If we have too few, add random points
            while len(weights) < N:
                w = np.random.default_rng(len(weights)).random(n_obj)
                w = w / np.sum(w)
                weights.append(w)
            return np.array(weights[:N])
        else:
            # General case: random weights on simplex
            rng = np.random.default_rng(0)
            weights = []
            for _ in range(N):
                w = rng.random(n_obj)
                w = w / np.sum(w)
                weights.append(w)
            return np.array(weights)
# EVOLVE-BLOCK-END
