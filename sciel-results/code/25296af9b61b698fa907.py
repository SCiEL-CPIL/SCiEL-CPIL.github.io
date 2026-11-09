"""Multi-objective optimizer on RE37 with structured search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS + adaptive local search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1

        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]

        fronts = [front]
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf

            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue

            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Latin Hypercube Sampling (80 evals)
        n_lhs = 80
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_lhs)
            X_lhs[:, d] = lo[d] + (perm + 0.5) * (hi[d] - lo[d]) / n_lhs

        F_all = [problem(x) for x in X_lhs]
        X_all = list(X_lhs)
        used = n_lhs

        if time.time() > deadline:
            return np.array(F_all)

        # Phase 2: Identify non-dominated points and refine
        F_arr = np.array(F_all)
        fronts = self._fast_non_dominated_sort(F_arr)
        front_indices = fronts[0] if fronts else list(range(len(F_arr)))

        # Select diverse subset for refinement (max 15 points)
        if len(front_indices) > 15:
            dist = self._crowding_distance(F_arr, front_indices)
            order = np.argsort(dist)[::-1]
            front_indices = [front_indices[i] for i in order[:15]]

        remaining = self.budget - used
        n_refine = len(front_indices)
        if n_refine == 0:
            return F_arr

        # Adaptive local search: for each seed point, try multiple scales
        # and directions, accepting improvements
        step_scales = [0.15, 0.1, 0.05, 0.025, 0.01]
        n_attempts_per_scale = max(1, remaining // (n_refine * len(step_scales)))

        for idx in front_indices:
            if used >= self.budget or time.time() > deadline - 5:
                break

            x_curr = X_all[idx].copy()
            f_curr = F_all[idx].copy()

            for scale in step_scales:
                if used >= self.budget or time.time() > deadline - 5:
                    break

                step_size = scale * span
                improved = False

                for _ in range(n_attempts_per_scale):
                    if used >= self.budget or time.time() > deadline - 5:
                        break

                    # Random direction perturbation
                    direction = np.random.standard_normal(self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    x_new = x_curr + step_size * direction
                    x_new = np.clip(x_new, lo, hi)

                    f_new = problem(x_new)
                    used += 1

                    # Check if new point is non-dominated by current
                    if np.all(f_new <= f_curr) and np.any(f_new < f_curr):
                        x_curr = x_new
                        f_curr = f_new
                        X_all.append(x_new)
                        F_all.append(f_new)
                        improved = True
                        break
                    else:
                        # Still record for diversity
                        X_all.append(x_new)
                        F_all.append(f_new)

                if improved:
                    continue  # Try next smaller scale
                else:
                    break  # No improvement at this scale, stop

        return np.array(F_all)
# EVOLVE-BLOCK-END
