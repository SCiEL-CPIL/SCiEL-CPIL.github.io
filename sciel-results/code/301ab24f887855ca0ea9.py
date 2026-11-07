"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for WFG4 with NSGA-II style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _crowding_distance(self, F):
        """Compute crowding distance for each point in F."""
        n = len(F)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F[order[-1], m] - F[order[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive of all non-dominated points found so far
        archive_X = X.copy()
        archive_F = F.copy()

        # Adaptive mutation parameters
        base_scale = 0.25
        min_scale = 0.01

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated front from current population
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Compute crowding distance for the non-dominated front
            nd_F = F[nd_indices]
            cd = self._crowding_distance(nd_F)

            # Select parent: prefer points with higher crowding distance
            # Use tournament selection on crowding distance
            if len(nd_indices) >= 2:
                i = nd_indices[rng.integers(len(nd_indices))]
                j = nd_indices[rng.integers(len(nd_indices))]
                # Map back to crowding distance
                i_pos = np.where(nd_indices == i)[0][0]
                j_pos = np.where(nd_indices == j)[0][0]
                if cd[i_pos] >= cd[j_pos]:
                    parent_idx = i
                else:
                    parent_idx = j
            else:
                parent_idx = nd_indices[0]

            # Adaptive mutation
            progress = used / self.budget
            scale = base_scale * (1 - progress) + min_scale * progress

            # Gaussian mutation
            mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update population: replace worst individual
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Update archive
            archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
