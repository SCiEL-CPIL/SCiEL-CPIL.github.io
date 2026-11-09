"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using a decomposition-based search with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        if self.budget > 1000:
            pop_size = 200

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        archive_X = X.copy()
        archive_F = F.copy()

        def non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    # j dominates i if j <= i in all objectives and j < i in at least one
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        def update_archive(new_x, new_f):
            """Add new solution to archive, maintaining non-dominance."""
            nonlocal archive_X, archive_F
            # Check if new solution is dominated by any in archive
            if np.any(np.all(archive_F <= new_f) and np.any(archive_F < new_f)):
                return
            # Remove archive members dominated by new solution
            keep = ~np.any(np.all(new_f <= archive_F) and np.any(new_f < archive_F), axis=1)
            archive_X = archive_X[keep]
            archive_F = archive_F[keep]
            # Add new solution
            archive_X = np.vstack([archive_X, new_x.reshape(1, -1)])
            archive_F = np.vstack([archive_F, new_f.reshape(1, -1)])

        # Weight vectors for decomposition
        n_weights = pop_size
        weights = np.array([i / (n_weights - 1) for i in range(n_weights)])
        if self.n_obj == 2:
            # For 2 objectives, weights are (w, 1-w)
            W = np.column_stack([weights, 1 - weights])
        else:
            # For more objectives, use simplex lattice
            W = rng.dirichlet(np.ones(self.n_obj), size=n_weights)

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # For each subproblem, select parents closest to weight direction
            child_X = np.empty_like(X)
            child_F = np.empty_like(F)

            for k in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select two parents: closest to weight k
                # Project objective values onto weight direction
                if self.n_obj == 2:
                    proj = F * W[k]
                else:
                    proj = F @ W[k]
                # Sort by projection (lower is better for minimization)
                order = np.argsort(proj)
                p1 = order[0]
                p2 = order[1] if len(order) > 1 else order[0]

                # Crossover: blend parents
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]

                # Mutation: Gaussian perturbation
                sigma = 0.1 * span
                child = child + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst individual in population
                # Use sum of normalized objectives as a simple fitness
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                worst_idx = np.argmax(F_norm.sum(axis=1))

                child_X[k] = child
                child_F[k] = fc

                # Update population
                X[worst_idx] = child
                F[worst_idx] = fc

                # Update archive
                update_archive(child, fc)

            # Periodically recompute non-dominated set
            if used % (pop_size * 2) < pop_size:
                pass  # Just continue

        return X
# EVOLVE-BLOCK-END
