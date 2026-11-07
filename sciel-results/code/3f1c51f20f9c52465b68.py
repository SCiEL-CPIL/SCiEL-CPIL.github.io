"""Multi-objective optimizer on WFG4 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II inspired multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, self.budget // 3)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        # Archive of all non-dominated points found so far
        archive_X = list(X)
        archive_F = list(F)

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (list of indices)."""
            n = len(F_arr)
            # domination count and set of dominated indices
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            nd = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
                else:
                    break

            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        def select_parents(pop_X, pop_F, n_parents):
            """Select parents using non-dominated sorting and crowding distance."""
            fronts = fast_non_dominated_sort(pop_F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_parents:
                    selected.extend(front)
                else:
                    cd = crowding_distance(pop_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:n_parents - len(selected)])
                    break
            return selected

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for d in range(self.dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
            return child1, child2

        def mutate(x, sigma=0.1):
            """Gaussian mutation."""
            return x + sigma * (hi - lo) * rng.standard_normal(self.dim)

        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Select parents
            n_parents = min(pop_size, len(archive_X))
            if n_parents < 2:
                n_parents = 2
            parent_indices = select_parents(
                np.array(archive_X[:n_parents]),
                np.array(archive_F[:n_parents]),
                n_parents
            )
            # Ensure we have enough parents
            while len(parent_indices) < n_parents:
                parent_indices.append(rng.integers(n_parents))

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size // 2, self.budget - evaluations)

            for _ in range(n_offspring):
                i1 = parent_indices[rng.integers(len(parent_indices))]
                i2 = parent_indices[rng.integers(len(parent_indices))]
                x1 = archive_X[i1]
                x2 = archive_X[i2]

                child1, child2 = crossover(x1, x2)
                child1 = mutate(child1)
                child2 = mutate(child2)

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                if evaluations < self.budget:
                    f1 = problem(child1)
                    evaluations += 1
                    offspring_X.append(child1)
                    offspring_F.append(f1)

                if evaluations < self.budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if len(offspring_X) == 0:
                break

            # Combine archive and offspring
            all_X = archive_X + offspring_X
            all_F = archive_F + offspring_F

            # Select next generation using non-dominated sorting
            n_select = min(pop_size, len(all_X))
            fronts = fast_non_dominated_sort(np.array(all_F))
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    cd = crowding_distance(np.array(all_F), front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:n_select - len(selected)])
                    break

            # Update archive with selected
            archive_X = [all_X[i] for i in selected]
            archive_F = [all_F[i] for i in selected]

        return archive_X
# EVOLVE-BLOCK-END
