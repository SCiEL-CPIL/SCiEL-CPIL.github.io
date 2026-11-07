"""Multi-objective WFG7 optimizer: NSGA-II with adaptive mutation and LHS init."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        # Use 20 to allow more generations (10 vs ~7) per cell findings
        pop_size = 20

        # Latin Hypercube initialization for better spread
        X = self._lhs(rng, pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best non-dominated set found so far
        best_nondom = self._nondominated(F)
        best_points = X[best_nondom]
        best_F = F[best_nondom]

        # Adaptive mutation scale
        mut_scale = 0.25

        while used < budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            # fronts[0] is the first (best) front, etc.

            # Crowding distance for the first front
            if len(fronts) > 0 and len(fronts[0]) > 1:
                cd = self._crowding_distance(F[fronts[0]])
            else:
                cd = np.ones(len(fronts[0])) if len(fronts[0]) > 0 else np.array([])

            # Select parents from first front using binary tournament on crowding distance
            parent1 = self._tournament(cd, rng)
            parent2 = self._tournament(cd, rng)
            p1 = X[fronts[0][parent1]]
            p2 = X[fronts[0][parent2]]

            # SBX-style crossover + adaptive mutation
            if rng.random() < 0.9:
                # Simulated binary crossover
                child = np.zeros(dim)
                for i in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / 3.0)
                        child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / 3.0)
                        child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
                # Adaptive mutation
                child = child + mut_scale * span * rng.standard_normal(dim)
            else:
                child = p1 + mut_scale * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update population: replace worst individual by sum of objectives
            # among dominated points, or worst overall
            dominated_mask = self._domination_mask(F)
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                worst_among_dominated = dominated_indices[np.argmax(np.sum(F[dominated_indices], axis=1))]
                X[worst_among_dominated] = child
                F[worst_among_dominated] = fc
            else:
                # All non-dominated: replace worst by sum
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

            # Update best non-dominated set
            current_nondom = self._nondominated(F)
            if len(current_nondom) > 0:
                new_points = X[current_nondom]
                new_F = F[current_nondom]
                # Merge and re-select non-dominated
                all_points = np.vstack([best_points, new_points])
                all_F = np.vstack([best_F, new_F])
                merged_nondom = self._nondominated(all_F)
                best_points = all_points[merged_nondom]
                best_F = all_F[merged_nondom]

            # Adapt mutation scale: shrink as we get closer to deadline
            progress = used / budget
            mut_scale = 0.25 * (1.0 - 0.7 * progress) + 0.02

        return X

    def _lhs(self, rng, n, dim, lo, hi):
        """Latin Hypercube Sampling."""
        X = np.zeros((n, dim))
        for i in range(dim):
            perm = rng.permutation(n)
            for j in range(n):
                u = (perm[j] + rng.random()) / n
                X[j, i] = lo[i] + u * (hi[i] - lo[i])
        return X

    def _nondominated(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        nondom = np.ones(n, dtype=bool)
        for i in range(n):
            if not nondom[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nondom[i] = False
                    break
        return np.where(nondom)[0]

    def _domination_mask(self, F):
        """Return mask of dominated points (True = dominated)."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return dominated

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts, each front is a list of indices."""
        n = len(F)
        if n == 0:
            return []
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [set() for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].add(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1

        front = [i for i in range(n) if domination_count[i] == 0]
        if front:
            fronts.append(front)

        k = 0
        while front and k < len(fronts):
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
            k += 1

        return fronts

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of non-dominated points."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            sorted_idx = np.argsort(F[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / f_range
        return cd

    def _tournament(self, cd, rng):
        """Binary tournament selection based on crowding distance."""
        i, j = rng.integers(0, len(cd), 2)
        return i if cd[i] >= cd[j] else j
# EVOLVE-BLOCK-END
