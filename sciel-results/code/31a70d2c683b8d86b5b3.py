"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance between diversity and computational cost
        pop_size = min(30, max(10, budget // 8))

        # Initialize population uniformly
        X = lo + span * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # WFG4 has a known Pareto front; try to seed near it
        # WFG4 Pareto front: f1 = x1, f2 = 1 - sqrt(f1), f3 = ... (depends on problem)
        # We'll rely on the evolutionary process to find good solutions

        while used < budget:
            # Check time
            if time.time() - start_time > deadline:
                break

            # Generate offspring using differential evolution
            offspring = []
            offspring_F = []

            # Number of offspring per generation
            n_offspring = min(pop_size, budget - used)

            for i in range(n_offspring):
                if time.time() - start_time > deadline:
                    break

                # Select three distinct parents
                idx = np.random.choice(pop_size, 3, replace=False)
                a, b, c = X[idx[0]], X[idx[1]], X[idx[2]]
                base = X[i % pop_size]

                # DE/rand/1 mutation
                F_factor = 0.5 + 0.5 * np.random.rand()
                v = a + F_factor * (b - c)

                # Crossover
                CR = 0.7
                u = np.random.rand(dim) < CR
                if not np.any(u):
                    u[np.random.randint(dim)] = True

                child = np.where(u, v, base)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                offspring.append(child)
                offspring_F.append(fc)
                used += 1

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_F = np.array(offspring_F)

            # Combine population and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F_comb)

            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(X_new)
                    if remaining > 0:
                        crowding = self._crowding_distance(F_comb[front])
                        order = np.argsort(crowding)[::-1][:remaining]
                        for idx in order:
                            X_new.append(X_comb[front[idx]])
                            F_new.append(F_comb[front[idx]])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return X

    def _non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each is list of indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        if not front:
            return [list(range(n))]
        fronts.append(front)

        current_front = front
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        n_obj = F.shape[1]
        distance = np.zeros(n)

        for m in range(n_obj):
            order = np.argsort(F[:, m])
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf

            f_range = F[order[-1], m] - F[order[0], m]
            if f_range == 0:
                continue

            for i in range(1, n - 1):
                distance[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range

        return distance
# EVOLVE-BLOCK-END
