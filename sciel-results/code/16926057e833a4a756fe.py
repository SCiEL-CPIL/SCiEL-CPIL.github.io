"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 5))
        # Ensure we don't exceed budget
        if pop_size * 2 > budget:
            pop_size = max(10, budget // 3)

        # Initial population
        rng = np.random.default_rng(42)
        pop = lo + (hi - lo) * rng.random((pop_size, n))
        fitness = np.array([problem(x) for x in pop], dtype=float)
        evaluations = pop_size

        # NSGA-II loop
        while evaluations < budget:
            # Create offspring
            offspring_size = min(pop_size, budget - evaluations)
            if offspring_size <= 0:
                break

            # Tournament selection for parents
            def tournament_select(pop, fitness, k=2):
                idxs = rng.integers(0, len(pop), size=k)
                best = idxs[0]
                for i in idxs[1:]:
                    if self._dominates(fitness[i], fitness[best]):
                        best = i
                return pop[best].copy()

            children = []
            for _ in range(offspring_size):
                p1 = tournament_select(pop, fitness)
                p2 = tournament_select(pop, fitness)
                # Simulated binary crossover
                child = self._sbx(p1, p2, lo, hi)
                # Polynomial mutation
                child = self._mutate(child, lo, hi)
                children.append(child)

            children = np.array(children)
            child_fitness = np.array([problem(x) for x in children], dtype=float)
            evaluations += offspring_size

            # Combine parents and children
            combined = np.vstack([pop, children])
            combined_fitness = np.vstack([fitness, child_fitness])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_fitness)

            # Crowding distance
            crowding = self._crowding_distance(combined_fitness, fronts)

            # Select next population
            next_pop_idx = []
            for front in fronts:
                if len(next_pop_idx) + len(front) <= pop_size:
                    next_pop_idx.extend(front)
                else:
                    # Sort by crowding distance in descending order
                    front_sorted = sorted(front, key=lambda i: crowding[i], reverse=True)
                    needed = pop_size - len(next_pop_idx)
                    next_pop_idx.extend(front_sorted[:needed])
                    break

            pop = combined[next_pop_idx]
            fitness = combined_fitness[next_pop_idx]

        return pop

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _sbx(self, p1, p2, lo, hi):
        """Simulated binary crossover."""
        eta = 20
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                else:
                    child[i] = p1[i]
            else:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
                else:
                    child[i] = p1[i]
        return np.clip(child, lo, hi)

    def _mutate(self, x, lo, hi, eta=20):
        """Polynomial mutation."""
        for i in range(len(x)):
            if rng.random() < 1.0 / len(x):
                u = rng.random()
                if u < 0.5:
                    xy = (2 * u + (x[i] - lo[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1))
                    x[i] = lo[i] + (hi[i] - lo[i]) * (1 - xy)
                else:
                    xy = (2 * (1 - u) + (hi[i] - x[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1))
                    x[i] = lo[i] + (hi[i] - lo[i]) * xy
        return x

    def _non_dominated_sort(self, fitness):
        """Return list of fronts, each front is a list of indices."""
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        fronts = []
        front0 = []

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                front0.append(i)

        if front0:
            fronts.append(front0)

        current_front = front0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def _crowding_distance(self, fitness, fronts):
        """Compute crowding distance for each index."""
        n = len(fitness)
        crowding = np.zeros(n)

        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowding[i] = float('inf')
                continue

            for m in range(self.n_obj):
                sorted_idx = sorted(front, key=lambda i: fitness[i][m])
                crowding[sorted_idx[0]] = float('inf')
                crowding[sorted_idx[-1]] = float('inf')
                min_val = fitness[sorted_idx[0]][m]
                max_val = fitness[sorted_idx[-1]][m]
                range_val = max_val - min_val
                if range_val < 1e-14:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    crowding[sorted_idx[k]] += (fitness[sorted_idx[k + 1]][m] - fitness[sorted_idx[k - 1]][m]) / range_val

        return crowding
# EVOLVE-BLOCK-END
