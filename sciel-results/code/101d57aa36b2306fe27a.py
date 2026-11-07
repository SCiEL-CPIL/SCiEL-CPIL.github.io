"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary search with crowding distance and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _nondominated_mask(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        # F[i] is dominated by F[j] if F[j] <= F[i] in all objectives and < in at least one
        # Vectorized: for each pair (i,j), check if j dominates i
        # F[j] <= F[i] for all obj: (F[j, :] <= F[i, :]).all(axis=1)
        # and F[j] < F[i] in at least one obj: (F[j, :] < F[i, :]).any(axis=1)
        # Build dominance matrix: D[i,j] = True if j dominates i
        # Shape: (n, n) where D[i,j] = 1 if j dominates i
        le = (F[:, None, :] <= F[None, :, :])  # (n, n, n_obj): le[i,j,k] = F[j,k] <= F[i,k]
        lt = (F[:, None, :] < F[None, :, :])   # (n, n, n_obj): lt[i,j,k] = F[j,k] < F[i,k]
        # D[i,j] = all(le[i,j,:]) & any(lt[i,j,:])
        D = le.all(axis=2) & lt.any(axis=2)
        # D[i,j] = True means j dominates i
        # i is dominated if any j dominates i (j != i)
        # np.any(D, axis=1) gives for each i if any j dominates i
        dominated = D.any(axis=1)
        return ~dominated

    def _crowding_distance(self, F, nd_mask):
        """Compute crowding distance for non-dominated points."""
        n = len(F)
        cd = np.zeros(n)
        nd_indices = np.where(nd_mask)[0]
        m = len(nd_indices)
        if m <= 2:
            cd[nd_indices] = np.inf
            return cd
        for k in range(self.n_obj):
            # Sort non-dominated points by objective k
            sorted_idx = nd_indices[np.argsort(F[nd_indices, k])]
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0], k]
            f_max = F[sorted_idx[-1], k]
            if f_max - f_min < 1e-15:
                continue
            for j in range(1, m - 1):
                cd[sorted_idx[j]] += (F[sorted_idx[j + 1], k] - F[sorted_idx[j - 1], k]) / (f_max - f_min)
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Initial population
        pop_size = min(40, budget // 2)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation step size
        step = 0.3

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated set
            nd_mask = self._nondominated_mask(F)
            if not np.any(nd_mask):
                nd_mask = np.ones(len(F), dtype=bool)

            # Use crowding distance for parent selection among non-dominated
            cd = self._crowding_distance(F, nd_mask)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) > 0:
                # Select parent with highest crowding distance
                parent_idx = nd_indices[np.argmax(cd[nd_indices])]
            else:
                parent_idx = rng.integers(len(F))

            # Adaptive mutation: occasionally larger steps for exploration
            if rng.random() < 0.1:
                local_step = step * 2.0
            else:
                local_step = step

            child = X[parent_idx].copy()
            # Gaussian mutation
            child += local_step * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives, or random if all similar)
            # Use a simple replacement: replace the point with highest sum of objectives
            # among dominated points, or the one with lowest crowding distance among non-dominated
            if np.any(~nd_mask):
                # Replace worst dominated point
                dominated_indices = np.where(~nd_mask)[0]
                worst_in_dominated = dominated_indices[np.argmax(F[dominated_indices].sum(axis=1))]
                X[worst_in_dominated] = child
                F[worst_in_dominated] = fc
            else:
                # All non-dominated: replace the one with lowest crowding distance
                worst_idx = nd_indices[np.argmin(cd[nd_indices])]
                X[worst_idx] = child
                F[worst_idx] = fc

            # Adaptive step: reduce if improvement, increase if stagnation
            # Simple heuristic: track best sum
            current_best = np.min(F.sum(axis=1))
            if used % 20 == 0:
                # Occasionally adjust step
                if rng.random() < 0.5:
                    step *= 0.9
                else:
                    step *= 1.1
                step = np.clip(step, 0.05, 0.8)

        return X
# EVOLVE-BLOCK-END
