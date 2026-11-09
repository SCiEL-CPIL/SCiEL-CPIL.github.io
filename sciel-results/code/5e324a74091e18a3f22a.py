"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size - smaller for 200 eval budget to allow more generations
        pop_size = 40

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        def fast_nondominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts and dominance ranks."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_count = np.zeros(n, dtype=int)  # n_count[i] = number of points dominating i
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_count[i] += 1

            # First front: points not dominated by any other
            front = [i for i in range(n) if n_count[i] == 0]
            ranks = np.full(n, -1, dtype=int)
            if front:
                fronts.append(front)
                for i in front:
                    ranks[i] = 0

            k = 0
            while k < len(fronts) and fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break

            return fronts, ranks

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            F_sub = F[indices]

            for m in range(self.n_obj):
                min_val = F_sub[:, m].min()
                max_val = F_sub[:, m].max()
                if max_val - min_val < 1e-10:
                    continue
                sorted_idx = np.argsort(F_sub[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (max_val - min_val)

            return distances

        def tournament_select(F, ranks, cd_map):
            """Tournament selection with crowding distance tie-breaking."""
            i, j = rng.integers(0, len(F), size=2)
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                # Same rank, use crowding distance
                cd_i = cd_map.get(i, 0)
                cd_j = cd_map.get(j, 0)
                if cd_i > cd_j:
                    return i
                elif cd_j > cd_i:
                    return j
                else:
                    return i if rng.random() < 0.5 else j

        def crossover(x1, x2):
            """SBX crossover."""
            if rng.random() < 0.9:
                beta = 20  # Distribution index
                child1 = np.empty_like(x1)
                child2 = np.empty_like(x2)
                for d in range(self.dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta1 = (2 * u) ** (1.0 / (beta + 1))
                    else:
                        beta1 = (2 * (1 - u)) ** (1.0 / (beta + 1))
                    child1[d] = 0.5 * ((1 + beta1) * x1[d] + (1 - beta1) * x2[d])
                    beta2 = 1.0 / beta1
                    child2[d] = 0.5 * ((1 + beta2) * x1[d] + (1 - beta2) * x2[d])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20
            child = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.8:  # Mutation probability
                    u = rng.random()
                    if u < 0.5:
                        val = (2 * u) ** (1.0 / (eta + 1))
                        delta = val - 1
                    else:
                        val = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        delta = 1 - val
                    # Scale by range
                    child[d] += delta * (hi[d] - lo[d]) * 0.1
            return np.clip(child, lo, hi)

        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts, ranks = fast_nondominated_sort(F)

            # Compute crowding distances for all fronts
            cd_map = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx, cd_val in zip(front, cd):
                    cd_map[idx] = cd_val

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break

                # Select parents using tournament selection
                p1 = tournament_select(F, ranks, cd_map)
                p2 = tournament_select(F, ranks, cd_map)

                # Crossover
                c1, c2 = crossover(X[p1], X[p2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                evaluations += 2

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts, ranks = fast_nondominated_sort(F_combined)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_combined, front)
                    sorted_indices = np.argsort(-cd)
                    for k in range(remaining):
                        idx = front[sorted_indices[k]]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
