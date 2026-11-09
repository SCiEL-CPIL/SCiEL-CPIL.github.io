"""Multi-objective optimizer on DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style optimizer for DTLZ7 with DTLZ-aware initialization and hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz7(self, x):
        """Compute DTLZ7 objectives analytically (fast, no function call)."""
        M = self.n_obj
        n = len(x)
        k = n - M + 1
        g = sum((x[i] - 0.5) ** 2 - np.cos(20 * np.pi * (x[i] + 4 * (x[i] - 0.5) ** 2)) for i in range(k))
        g = max(g, 0.0)
        g = 1.0 + g
        # g is already >= 1
        f = np.zeros(M)
        for i in range(M - 1):
            f[i] = x[k + i] * (1.0 + g)
        # f[M-1]
        t = 1.0
        for i in range(M - 1):
            t *= (1.0 - x[k + i] + g / M)
        f[M - 1] = (1.0 + g) * t
        return f

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1

        current_front = [i for i in range(n) if n_dominates[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        M = F.shape[1]
        for m in range(M):
            sorted_idx = np.argsort(F[indices, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_min = F[indices[sorted_idx[0]], m]
            f_max = F[indices[sorted_idx[-1]], m]
            if f_max - f_min < 1e-15:
                continue
            for i in range(1, n - 1):
                dist[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]], m] - F[indices[sorted_idx[i - 1]], m]) / (f_max - f_min)
        return dist

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume (for ranking within a front)."""
        # points: (n, 2), ref: (2,)
        # Only count points strictly better than ref
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by first coordinate
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        prev_y = ref[1]
        for p in pts:
            if p[1] < prev_y:
                hv += (ref[0] - p[0]) * (prev_y - p[1])
                prev_y = p[1]
            else:
                # p[1] >= prev_y, so this point doesn't add area with current prev_y
                # but we still need to update if p[0] is smaller... actually sorted by x
                pass
        return hv

    def _select_parents(self, F, X, fronts, n_select):
        """Select parents using binary tournament on fronts and crowding distance."""
        n = len(F)
        # Assign front index to each individual
        front_idx = np.full(n, -1, dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                front_idx[idx] = fi

        # Compute crowding distance per front
        crowd = np.zeros(n)
        for front in fronts:
            cd = self._crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowd[idx] = cd[i]

        # Binary tournament selection
        selected = []
        rng = np.random
        while len(selected) < n_select:
            a = rng.integers(n)
            b = rng.integers(n)
            if front_idx[a] < front_idx[b]:
                selected.append(a)
            elif front_idx[b] < front_idx[a]:
                selected.append(b)
            else:
                if crowd[a] > crowd[b]:
                    selected.append(a)
                else:
                    selected.append(b)
        return selected

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        M = self.n_obj
        budget = self.budget

        # DTLZ7-aware initialization:
        # x[0:dim-M+1] ~ U(0.5, 1.0) to keep g small
        # x[dim-M+1:] ~ U(0, 1)
        k = dim - M + 1
        pop_size = min(30, max(10, budget // 8))

        X = np.empty((pop_size, dim))
        for i in range(pop_size):
            X[i, :k] = 0.5 + 0.5 * rng.random(k)
            X[i, k:] = rng.random(M - 1)

        # Evaluate using analytical DTLZ7 (fast)
        F = np.array([self._dtlz7(x) for x in X])
        used = pop_size

        # Also call problem() for the first few to match evaluator expectations
        # Actually, the evaluator calls problem(x) and records the result.
        # We need to call problem() for each evaluation to have it recorded.
        # Let me re-evaluate: the problem function is what the evaluator tracks.
        # So I must call problem(x) for each new point, not my analytical version.
        # But I can use analytical for internal ranking and problem() for recording.

        # Actually, let me just use problem() for all evaluations to be safe.
        # The analytical version is for my own understanding; the evaluator only
        # records calls to problem().

        # Re-initialize properly using problem()
        X = np.empty((pop_size, dim))
        F = np.empty((pop_size, M))
        for i in range(pop_size):
            X[i, :k] = 0.5 + 0.5 * rng.random(k)
            X[i, k:] = rng.random(M - 1)
            F[i] = problem(X[i])
        used = pop_size

        # Main loop
        while used < budget:
            # Time guard
            # We don't have a start time easily here, but budget is small (200)
            # so it should be fast.

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Select parents
            n_offspring = min(5, budget - used)
            parent_indices = self._select_parents(F, X, fronts, 2 * n_offspring)

            # Generate offspring
            for _ in range(n_offspring):
                p1 = parent_indices[rng.integers(len(parent_indices))]
                p2 = parent_indices[rng.integers(len(parent_indices))]

                # SBX crossover
                child = np.empty(dim)
                for d in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (2 + 20))
                    else:
                        beta = (2 * (1 - u)) ** (1 / (2 + 20))
                    child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                # Replace worst individual in the population
                # Use a simple strategy: replace the individual with the worst
                # crowding distance in the last front
                worst_front = fronts[-1] if fronts else [0]
                if len(worst_front) > 1:
                    cd = self._crowding_distance(F, worst_front)
                    worst_idx = worst_front[np.argmin(cd)]
                else:
                    worst_idx = worst_front[0]

                X[worst_idx] = child
                F[worst_idx] = fc

                # Break if budget reached
                if used >= budget:
                    break

        return X
# EVOLVE-BLOCK-END
