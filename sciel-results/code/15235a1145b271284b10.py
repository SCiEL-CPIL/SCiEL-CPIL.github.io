# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ1 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        pop_size = min(92, max(40, budget // 3))

        # Initial population: stratified sampling for better coverage
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # Use a mix of random and structured initialization
            if i % 2 == 0:
                X[i] = lo + (hi - lo) * rng.random(dim)
            else:
                # Structured: spread across dimensions
                X[i] = lo + (hi - lo) * ((i + 1) / (pop_size + 1)) * rng.random(dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 1.0 / dim
        sigma = 0.1  # mutation strength

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)

            # Build offspring
            offspring_X = []
            offspring_F = []

            # Tournament selection and reproduction
            while len(offspring_X) < pop_size:
                # Check deadline
                if time.time() > deadline:
                    break

                # Tournament selection
                p1 = self._tournament_select(X, F, fronts, rng)
                p2 = self._tournament_select(X, F, fronts, rng)

                # Simulated binary crossover
                if rng.random() < crossover_rate:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, sigma, mutation_rate, dim, rng)
                c2 = self._polynomial_mutation(c2, lo, hi, sigma, mutation_rate, dim, rng)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1

                if used >= budget:
                    break

                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined population
            combined_fronts = self._non_dominated_sort(combined_F)

            # Select next generation
            X, F = self._select_next_generation(combined_X, combined_F, combined_fronts, pop_size, rng)

        return X

    def _non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each front is a list of indices)."""
        n = len(F)
        if n == 0:
            return []

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
        if front:
            fronts.append(front)

        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        n_obj = F.shape[1]

        for k in range(n_obj):
            obj_vals = F[indices, k]
            sorted_idx = np.argsort(obj_vals)
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
            if obj_range > 1e-12:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range

        return distances

    def _tournament_select(self, X, F, fronts, rng, tournament_size=2):
        """Binary tournament selection."""
        i = rng.integers(len(X))
        j = rng.integers(len(X))

        # Find front indices
        front_i = next(idx for idx, front in enumerate(fronts) if i in front)
        front_j = next(idx for idx, front in enumerate(fronts) if j in front)

        if front_i < front_j:
            return X[i].copy()
        elif front_j < front_i:
            return X[j].copy()
        else:
            # Same front: use crowding distance
            front_indices = fronts[front_i]
            cd = self._crowding_distance(F, front_indices)
            cd_i = cd[front_indices.index(i)]
            cd_j = cd[front_indices.index(j)]
            return X[i].copy() if cd_i >= cd_j else X[j].copy()

    def _sbx_crossover(self, p1, p2, lo, hi, rng):
        """Simulated binary crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = (2 * min(p1[i], p2[i]) - max(p1[i], p2[i]) * 0 + max(p1[i], p2[i])) / (2 * min(p1[i], p2[i]) - max(p1[i], p2[i]) * 0 + max(p1[i], p2[i])) if min(p1[i], p2[i]) != max(p1[i], p2[i]) else 1.0
                        # Simplified SBX
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        c1[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                        c2[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, sigma, rate, dim, rng):
        """Polynomial mutation."""
        x = x.copy()
        for i in range(dim):
            if rng.random() < rate:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (sigma + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (sigma + 1))
                x[i] += delta * (hi[i] - lo[i])
        return np.clip(x, lo, hi)

    def _select_next_generation(self, X, F, fronts, pop_size, rng):
        """Select next generation using non-dominated sorting and crowding distance."""
        selected_X = []
        selected_F = []
        selected_indices = []

        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
            else:
                # Need to truncate this front
                remaining = pop_size - len(selected_indices)
                cd = self._crowding_distance(F, front)
                sorted_front = sorted(front, key=lambda idx: cd[front.index(idx)], reverse=True)
                selected_indices.extend(sorted_front[:remaining])
                break

        selected_X = X[selected_indices]
        selected_F = F[selected_indices]

        return selected_X, selected_F
# EVOLVE-BLOCK-END
