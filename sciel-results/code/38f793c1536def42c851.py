# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = list(zip(X, F))

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination relations
            n_dominated = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1

            current_front = [i for i in range(n) if n_dominated[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Add any remaining (shouldn't happen with correct domination)
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return cd

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate remaining if not yet done
            if used < budget:
                # Create offspring
                # Use SBX crossover and polynomial mutation
                offspring = []
                while len(offspring) < pop_size:
                    # Select parents using tournament
                    p1_idx = self._tournament_select(F, pop_size, rng)
                    p2_idx = self._tournament_select(F, pop_size, rng)
                    p1 = X[p1_idx]
                    p2 = X[p2_idx]

                    # SBX crossover
                    if rng.random() < 0.9:
                        c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                    else:
                        c1 = p1.copy()
                        c2 = p2.copy()

                    # Polynomial mutation
                    c1 = self._poly_mutation(c1, lo, hi, dim, rng)
                    c2 = self._poly_mutation(c2, lo, hi, dim, rng)

                    offspring.append(c1)
                    if len(offspring) < pop_size:
                        offspring.append(c2)

                offspring = np.array(offspring[:pop_size])
                F_off = np.array([problem(x) for x in offspring])
                used += len(offspring)

                # Combine parents and offspring
                X_all = np.vstack([X, offspring])
                F_all = np.vstack([F, F_off])

                # Non-dominated sort
                fronts = non_dominated_sort(F_all)

                # Select next population
                X_next = []
                F_next = []
                for front in fronts:
                    if len(X_next) + len(front) <= pop_size:
                        for idx in front:
                            X_next.append(X_all[idx])
                            F_next.append(F_all[idx])
                    else:
                        # Need to select from this front using crowding distance
                        cd = crowding_distance(F_all, front)
                        order = np.argsort(cd)[::-1]  # descending
                        remaining = pop_size - len(X_next)
                        for i in range(remaining):
                            idx = front[order[i]]
                            X_next.append(X_all[idx])
                            F_next.append(F_all[idx])
                        break

                X = np.array(X_next)
                F = np.array(F_next)

                # Track all points
                for x, f in zip(offspring, F_off):
                    all_points.append((x, f))

        return X

    def _tournament_select(self, F, pop_size, rng, k=2):
        """Tournament selection based on non-domination rank."""
        # Simple: pick k random, return the one with best (lowest) sum of objectives
        # as a proxy for rank (faster than full non-dominated sort each time)
        indices = rng.integers(0, pop_size, size=k)
        best = indices[0]
        best_val = np.sum(F[best])
        for idx in indices[1:]:
            val = np.sum(F[idx])
            if val < best_val:
                best = idx
                best_val = val
        return best

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20):
        """Simulated Binary Crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi, dim, rng, eta=20):
        """Polynomial mutation."""
        for i in range(dim):
            if rng.random() < 1.0 / dim:
                u = rng.random()
                x_range = hi[i] - lo[i]
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                x[i] += delta * x_range
        return np.clip(x, lo, hi)
# EVOLVE-BLOCK-END
