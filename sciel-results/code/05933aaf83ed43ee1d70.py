"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Random initialization
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to given indices."""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for obj in range(n_obj):
                vals = F[indices, obj]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 0:
                    for i in range(1, m - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return cd

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Build next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                else:
                    # Need to select from this front
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F, np.array(front))
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size - used
            if n_offspring <= 0:
                break

            for _ in range(min(n_offspring, pop_size)):
                if time.time() > deadline:
                    break
                # Select two parents using tournament
                p1_idx, p2_idx = self._tournament_select(F, pop_size)
                # Crossover (SBX-like with simple blend)
                if rng.random() < 0.9:
                    child1, child2 = self._crossover(X[p1_idx], X[p2_idx], lo, hi, rng)
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()
                # Mutation
                child1 = self._mutate(child1, lo, hi, rng)
                child2 = self._mutate(child2, lo, hi, rng)

                fc1 = problem(child1)
                used += 1
                offspring_X.append(child1)
                offspring_F.append(fc1)

                if used < budget and len(offspring_X) < pop_size:
                    fc2 = problem(child2)
                    used += 1
                    offspring_X.append(child2)
                    offspring_F.append(fc2)

            if not offspring_X:
                break

            # Combine and select
            all_X = np.vstack([X, np.array(offspring_X)])
            all_F = np.vstack([F, np.array(offspring_F)])

            fronts = fast_non_dominated_sort(all_F)

            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(all_F, np.array(front))
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _tournament_select(self, F, pop_size):
        """Binary tournament selection based on dominance."""
        i = np.random.randint(0, pop_size)
        j = np.random.randint(0, pop_size)
        # Check if i dominates j
        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
            return i, j
        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
            return j, i
        else:
            return i, j

    def _crossover(self, p1, p2, lo, hi, rng):
        """Simple blend crossover."""
        alpha = rng.random()
        child1 = alpha * p1 + (1 - alpha) * p2
        child2 = (1 - alpha) * p1 + alpha * p2
        child1 = np.clip(child1, lo, hi)
        child2 = np.clip(child2, lo, hi)
        return child1, child2

    def _mutate(self, x, lo, hi, rng):
        """Gaussian mutation."""
        span = hi - lo
        sigma = 0.1
        mutation = rng.standard_normal(x.shape) * sigma * span
        x_new = x + mutation
        return np.clip(x_new, lo, hi)
# EVOLVE-BLOCK-END
