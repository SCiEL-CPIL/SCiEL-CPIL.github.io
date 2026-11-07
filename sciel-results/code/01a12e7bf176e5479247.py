"""Decomposition-based MOEA/D for WFG9 with improved exploration and exploitation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D-style decomposition multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: larger for better coverage, but allow multiple generations
        pop_size = min(80, max(30, budget // 4))
        if pop_size * 2 > budget:
            pop_size = max(10, budget // 2)

        # Generate weight vectors using simplex-lattice or random normalized weights
        weights = self._generate_weights(n_obj, pop_size, rng)

        # Ideal and nadir points for normalization (updated dynamically)
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)

        # Initialize population with stratified sampling for better coverage
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            # Mix of random and stratified initialization
            if i % 3 == 0:
                # Stratified: split each dimension into 3 parts
                x = np.zeros(dim)
                for d in range(dim):
                    part = (i % 3)
                    x[d] = lo[d] + span[d] * (part + rng.random()) / 3.0
                X[i] = x
            else:
                X[i] = lo + span * rng.random(dim)
            F[i] = problem(X[i])
        used = pop_size

        # Update ideal/nadir
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        # Avoid division by zero in normalization
        denom = nadir - ideal
        denom[denom < 1e-10] = 1.0

        # Track best solutions found so far (archive)
        archive_F = F.copy()
        archive_X = X.copy()

        # Neighborhood size for MOEA/D
        T = max(3, pop_size // 4)

        # Precompute neighbors
        neighbors = self._compute_neighbors(weights, T)

        # Adaptive mutation sigma
        sigma_start = 0.15
        sigma_end = 0.02
        total_iters = budget - pop_size

        while used < budget:
            # Select subproblem index with some bias toward under-explored regions
            idx = rng.integers(pop_size)
            w = weights[idx]

            # Get neighbor indices
            nbrs = neighbors[idx]

            # Select parent from neighbors using fitness
            fitnesses = np.zeros(len(nbrs))
            for i, ni in enumerate(nbrs):
                f = F[ni]
                # Normalized objectives
                fn = (f - ideal) / denom
                fn = np.clip(fn, 0.0, 1.0)
                # Tchebycheff aggregation
                fitnesses[i] = np.max(w * fn)

            # Select parent: best with probability, otherwise random
            best_local = np.argmin(fitnesses)
            if rng.random() < 0.75:
                parent_idx = nbrs[best_local]
            else:
                parent_idx = nbrs[rng.integers(len(nbrs))]

            # Adaptive mutation: decrease sigma over time
            progress = used / budget
            sigma = sigma_start + (sigma_end - sigma_start) * progress
            
            # Mutation: Gaussian perturbation with adaptive rate
            mutation_rate = 0.4 + 0.2 * (1 - progress)  # Higher mutation early
            child = X[parent_idx].copy()
            mutation_mask = rng.random(dim) < mutation_rate
            if mutation_mask.any():
                child[mutation_mask] += sigma * span[mutation_mask] * rng.standard_normal(mutation_mask.sum())
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            archive_F = np.vstack([archive_F, [fc]])
            archive_X = np.vstack([archive_X, [child]])

            # Decide whether to replace X[idx]
            # Compare child with current solution at idx using Tchebycheff
            fn_child = (fc - ideal) / denom
            fn_child = np.clip(fn_child, 0.0, 1.0)
            fit_child = np.max(w * fn_child)

            fn_curr = (F[idx] - ideal) / denom
            fn_curr = np.clip(fn_curr, 0.0, 1.0)
            fit_curr = np.max(w * fn_curr)

            if fit_child <= fit_curr:
                X[idx] = child
                F[idx] = fc

            # Update ideal and nadir
            ideal = np.minimum(ideal, fc)
            nadir = np.maximum(nadir, fc)
            denom = nadir - ideal
            denom[denom < 1e-10] = 1.0

        # Return the archive (all evaluated points)
        return archive_X

    def _generate_weights(self, n_obj, pop_size, rng):
        """Generate weight vectors that sum to 1 and are non-negative using stratified simplex sampling."""
        weights = []
        if n_obj == 3:
            # Generate points on the 2D simplex using stratified sampling
            # Use a grid with refinement
            n_per_edge = max(2, int(np.sqrt(pop_size / 2)))
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    k = n_per_edge - i - j
                    w1 = i / n_per_edge
                    w2 = j / n_per_edge
                    w3 = k / n_per_edge
                    # Add small random perturbation to break ties
                    w = np.array([w1, w2, w3])
                    w += rng.random(3) * 0.02
                    w = np.clip(w, 0.01, None)
                    w = w / w.sum()
                    weights.append(w)
            
            # If we have too many, subsample
            if len(weights) > pop_size:
                indices = rng.choice(len(weights), pop_size, replace=False)
                weights = [weights[i] for i in indices]
            
            # If too few, add random weights
            while len(weights) < pop_size:
                w = rng.dirichlet(np.ones(n_obj))
                weights.append(w)
        else:
            weights = [rng.dirichlet(np.ones(n_obj)) for _ in range(pop_size)]

        weights = np.array(weights, dtype=float)
        # Normalize to sum to 1
        sums = weights.sum(axis=1, keepdims=True)
        weights = weights / sums
        return weights

    def _compute_neighbors(self, weights, T):
        """Compute T nearest neighbors for each weight vector based on Euclidean distance."""
        n = len(weights)
        neighbors = []
        for i in range(n):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            # Exclude self
            dists[i] = np.inf
            nbr_indices = np.argsort(dists)[:T]
            neighbors.append(nbr_indices.tolist())
        return neighbors
# EVOLVE-BLOCK-END
