import numpy as np


class Optimizer:
    """Multi-objective optimizer using Differential Evolution with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[indices, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F[indices[order[-1]], m] - F[indices[order[0]], m]
            if f_range > 0:
                for k in range(1, n - 1):
                    dist[order[k]] += (F[indices[order[k + 1]], m] - F[indices[order[k - 1]], m]) / f_range
        return dist

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population: 20 points via LHS-like stratified sampling
        pop_size = 20
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            X[:, d] = np.linspace(lo[d], hi[d], pop_size, endpoint=False) + rng.random(pop_size) * (hi[d] - lo[d]) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # External archive
        archive_X = X.copy()
        archive_F = F.copy()
        archive_idx = np.arange(pop_size)

        def update_archive(X_new, F_new):
            nonlocal archive_X, archive_F, archive_idx
            # Add new points to archive candidates
            X_all = np.vstack([archive_X, X_new])
            F_all = np.vstack([archive_F, F_new])
            # Remove dominated points
            n = len(F_all)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_all[i] <= F_all[j]) and np.any(F_all[i] < F_all[j]):
                        dominated[j] = True
            non_dom_idx = np.where(~dominated)[0]
            # If too many, use crowding to trim
            if len(non_dom_idx) > 50:
                cd = self._crowding_distance(F_all, non_dom_idx)
                # Keep points with highest crowding distance
                order = np.argsort(cd)[::-1][:50]
                non_dom_idx = non_dom_idx[order]
            archive_X = X_all[non_dom_idx]
            archive_F = F_all[non_dom_idx]
            archive_idx = non_dom_idx

        update_archive(X, F)

        # DE parameters
        CR = 0.8
        F_DE = 0.5

        while used < self.budget:
            # Select 3 distinct parents from archive
            if len(archive_idx) < 3:
                break
            i1 = rng.integers(len(archive_idx))
            i2 = rng.integers(len(archive_idx))
            i3 = rng.integers(len(archive_idx))
            # Ensure distinct
            attempts = 0
            while (i1 == i2 or i1 == i3 or i2 == i3) and attempts < 10:
                i2 = rng.integers(len(archive_idx))
                i3 = rng.integers(len(archive_idx))
                attempts += 1
            if i1 == i2 or i1 == i3 or i2 == i3:
                break

            x1 = archive_X[i1]
            x2 = archive_X[i2]
            x3 = archive_X[i3]

            # DE/rand/1/bin mutation
            v = x1 + F_DE * (x2 - x3)
            v = np.clip(v, lo, hi)

            # Crossover
            j_rand = rng.integers(self.dim)
            mask = rng.random(self.dim) < CR
            mask[j_rand] = True
            u = np.where(mask, v, x1)

            fu = problem(u)
            used += 1

            # Update archive with new point
            update_archive(u.reshape(1, -1), fu.reshape(1, -1))

        return archive_X
