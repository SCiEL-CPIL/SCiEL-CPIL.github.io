"""Decomposition-based MOEA with Tchebycheff scalarization for WFG9."""
import numpy as np
import time


class Optimizer:
    """Tchebycheff decomposition with local search and global archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, n_obj, num_points):
        """Generate weight vectors on the simplex."""
        weights = []
        if n_obj == 3:
            # Grid on 2-simplex
            steps = int(np.ceil(num_points ** 0.5))
            for i in range(steps + 1):
                for j in range(steps + 1 - i):
                    k = steps - i - j
                    w = np.array([i, j, k], dtype=float) / steps
                    weights.append(w)
            # Deduplicate
            unique = []
            for w in weights:
                if not any(np.allclose(w, u) for u in unique):
                    unique.append(w)
            return np.array(unique)
        else:
            # Fallback: random weights
            rng = np.random.default_rng(42)
            W = rng.random((num_points, n_obj))
            W /= W.sum(axis=1, keepdims=True)
            return W

    def _tchebycheff(self, f, w, z):
        """Tchebycheff scalarized value: max_j w_j * (f_j - z_j)."""
        return np.max(w * (f - z))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Generate weight vectors
        num_weights = min(20, budget // 10)
        weights = self._generate_weights(n_obj, num_weights)
        n_sub = len(weights)

        # Initialize: random solutions for each subproblem
        X = lo + (hi - lo) * rng.random((n_sub, dim))
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Ideal point (min of each objective so far)
        z = F.min(axis=0).copy()
        # Add small epsilon to avoid zero weights issues
        z = z - 1e-6

        # Scalarized values for each subproblem
        S = np.array([self._tchebycheff(F[i], weights[i], z) for i in range(n_sub)])

        # Global archive: list of (x, f) non-dominated solutions
        archive_X = [X[i].copy() for i in range(n_sub)]
        archive_F = [F[i].copy() for i in range(n_sub)]

        # Local search parameters
        neighborhood_size = 3
        max_local_steps = 2

        def is_dominated(f_new, archive_F_list):
            """Check if f_new is dominated by any in archive."""
            for fa in archive_F_list:
                if np.all(fa <= f_new) and np.any(fa < f_new):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            """Add (x, f) to archive if non-dominated, remove dominated."""
            if is_dominated(f, archive_F):
                return
            # Remove dominated solutions from archive
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if np.all(af <= f) and np.any(af < f):
                    continue  # dominated by new
                new_X.append(ax)
                new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X[:] = new_X
            archive_F[:] = new_F

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Pick subproblem to work on (round-robin or random)
            idx = rng.integers(n_sub)

            # Get neighbors by weight distance
            dists = np.array([np.linalg.norm(weights[idx] - weights[j]) for j in range(n_sub)])
            neighbors = np.argsort(dists)[:neighborhood_size + 1][1:]  # exclude self

            # Local search: try mutating from best neighbor
            improved = False
            for _ in range(max_local_steps):
                if used >= budget or time.time() > deadline:
                    break

                # Pick a neighbor
                nidx = neighbors[rng.integers(len(neighbors))]

                # Mutate: blend parent with neighbor + noise
                child = X[idx].copy()
                # Crossover-like: blend
                alpha = rng.random()
                child = (1 - alpha) * child + alpha * X[nidx]
                # Add Gaussian noise
                sigma = 0.1 * (hi - lo)
                child = child + sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update ideal point
                z_new = np.minimum(z, fc)
                z = z_new

                # Check if child improves scalarized value for subproblem idx
                s_child = self._tchebycheff(fc, weights[idx], z)
                if s_child < S[idx] + 1e-10:
                    X[idx] = child
                    F[idx] = fc
                    S[idx] = s_child
                    improved = True

                # Update global archive
                update_archive(child, fc, archive_X, archive_F)

            # If no improvement, re-seed from archive
            if not improved and len(archive_X) > 0:
                # Pick a random archive solution
                aidx = rng.integers(len(archive_X))
                X[idx] = archive_X[aidx].copy()
                F[idx] = archive_F[aidx].copy()
                S[idx] = self._tchebycheff(F[idx], weights[idx], z)

        return X
