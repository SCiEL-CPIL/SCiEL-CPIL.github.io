"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer (MOEA/D) with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(budget, 40)
        if pop_size < 5:
            pop_size = 5

        # Weight vectors for MOEA/D
        # Generate weight vectors on the simplex
        weights = self._generate_weights(pop_size, n_obj)

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best found
        best_F = F.copy()

        # Neighbor structure: for each individual, find k nearest weight vectors
        k_neighbors = min(5, pop_size - 1)
        neighbors = self._compute_neighbors(weights, k_neighbors)

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents from neighbors
                nbrs = neighbors[i]
                # Pick 2 parents from neighbors (or from whole pop if small)
                if len(nbrs) >= 2:
                    p1_idx, p2_idx = rng.choice(nbrs, size=2, replace=False)
                else:
                    p1_idx = rng.integers(pop_size)
                    p2_idx = rng.integers(pop_size)

                # Crossover: SBX-like blend
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                # Blend with some randomness
                alpha = rng.random(dim)
                child = alpha * x1 + (1 - alpha) * x2

                # Mutation: Gaussian with adaptive step
                # Occasionally do a large jump
                if rng.random() < 0.1:
                    # Large jump: restart-like
                    jump_scale = 0.5
                else:
                    jump_scale = 0.15

                child = child + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update: replace individual i if child dominates it,
                # or if child is better in terms of decomposition
                if self._dominates(fc, F[i]):
                    X[i] = child
                    F[i] = fc
                else:
                    # Use decomposition with weight vector
                    # Tchebycheff-style: check if child is better for this weight
                    w = weights[i]
                    # Normalize objectives for fair comparison
                    # Simple approach: weighted sum
                    score_child = np.dot(w, fc)
                    score_parent = np.dot(w, F[i])
                    if score_child < score_parent:
                        X[i] = child
                        F[i] = fc

                # Track best
                if self._dominates(fc, best_F[i]):
                    best_F[i] = fc

        return X

    def _generate_weights(self, pop_size, n_obj):
        """Generate weight vectors on the simplex."""
        if n_obj == 1:
            return np.ones((pop_size, 1))

        # Use quasi-random sequence (Hammersley) for good spread
        weights = []
        # Generate points on the simplex
        # For 3 objectives, use a grid-like approach
        if n_obj == 3:
            # Distribute points on the 2D simplex
            n_per_edge = max(2, int(np.sqrt(pop_size)))
            for i in range(n_per_edge):
                for j in range(n_per_edge - i):
                    k = n_per_edge - i - j
                    w = np.array([i / (n_per_edge - 1), j / (n_per_edge - 1), k / (n_per_edge - 1)])
                    w = w / (np.sum(w) + 1e-10)
                    weights.append(w)
                    if len(weights) >= pop_size:
                        break
                if len(weights) >= pop_size:
                    break

        if len(weights) < pop_size:
            # Fill remaining with random points
            while len(weights) < pop_size:
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)

        weights = np.array(weights[:pop_size])
        # Ensure they are properly normalized
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-10)
        return weights

    def _compute_neighbors(self, weights, k):
        """Compute k nearest neighbors for each weight vector."""
        n = len(weights)
        # Compute pairwise distances
        diff = weights[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists, np.inf)
        neighbors = []
        for i in range(n):
            idx = np.argsort(dists[i])[:k]
            neighbors.append(idx)
        return neighbors

    def _dominates(self, a, b):
        """Check if a dominates b (all objectives <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)
# EVOLVE-BLOCK-END
