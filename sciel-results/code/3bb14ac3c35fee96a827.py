"""NSGA-II multi-objective optimizer for WFG9."""
import numpy as np
import time


class Optimizer:
    """NSGA-II with SBX crossover and polynomial mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = solutions dominated by i
        n_domin = [0] * n  # number of solutions that dominate i
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_domin[i] += 1

        current_front = [i for i in range(n) if n_domin[i] == 0]
        fronts.append(current_front)

        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_domin[j] -= 1
                    if n_domin[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for solutions in indices."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n

        dist = [0.0] * n
        n_obj = F.shape[1]

        for m in range(n_obj):
            obj_vals = F[indices, m]
            min_val = obj_vals.min()
            max_val = obj_vals.max()
            if max_val - min_val < 1e-14:
                dist[:] = [float('inf')] * n
                continue

            # Sort by objective m
            sorted_idx = np.argsort(obj_vals)
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')

            for k in range(1, n - 1):
                dist[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)

        return dist

    def _sbx_crossover(self, x1, x2, lo, hi, eta=20.0, prob=0.9):
        """Simulated binary crossover."""
        dim = len(x1)
        child1 = x1.copy()
        child2 = x2.copy()

        if rng.random() > prob:
            return child1, child2

        for i in range(dim):
            if rng.random() < 0.5:
                continue
            if abs(x1[i] - x2[i]) < 1e-14:
                continue

            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

            c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
            c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

            if c1 < lo[i] or c1 > hi[i]:
                c1 = x1[i]
            if c2 < lo[i] or c2 > hi[i]:
                c2 = x2[i]

            child1[i] = c1
            child2[i] = c2

        return child1, child2

    def _polynomial_mutation(self, x, lo, hi, eta=20.0, prob=0.1):
        """Polynomial mutation."""
        dim = len(x)
        child = x.copy()

        for i in range(dim):
            if rng.random() < prob:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))

                child[i] = x[i] + delta * (hi[i] - lo[i])
                child[i] = np.clip(child[i], lo[i], hi[i])

        return child

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        global rng
        rng = np.random.default_rng(0)

        # Population size
        pop_size = 50

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if used + 1 > budget or time.time() > deadline:
                    break

                # Select two parents using tournament selection
                p1 = self._tournament_select(X, F, k=2)
                p2 = self._tournament_select(X, F, k=2)

                # Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta=20.0, prob=0.9)

                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, eta=20.0, prob=1.0 / dim)
                c2 = self._polynomial_mutation(c2, lo, hi, eta=20.0, prob=1.0 / dim)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1

                if used >= budget or time.time() > deadline:
                    break

                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            front_idx = 0

            while front_idx < len(fronts) and len(next_X) < pop_size:
                front = fronts[front_idx]
                if len(next_X) + len(front) <= pop_size:
                    for i in front:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                    front_idx += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = self._crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for idx in sorted_front[:remaining]:
                        i = front[idx]
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                    break

            if len(next_X) == 0:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _tournament_select(self, X, F, k=2):
        """Tournament selection based on non-dominated rank and crowding distance."""
        n = len(X)
        if n <= k:
            idx = rng.integers(n)
            return X[idx].copy()

        # Get ranks
        fronts = self._fast_non_dominated_sort(F)
        rank = np.zeros(n, dtype=int)
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r

        # Get crowding distances
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 2:
                front_cd = self._crowding_distance(F, front)
                for j, i in enumerate(front):
                    cd[i] = front_cd[j]
            else:
                for i in front:
                    cd[i] = float('inf')

        # Tournament
        candidates = rng.choice(n, size=k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if rank[c] < rank[best]:
                best = c
            elif rank[c] == rank[best] and cd[c] > cd[best]:
                best = c

        return X[best].copy()
