"""Multi-objective optimizer for WFG9 using NSGA-II with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, budget)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 0.1
        sigma = 0.2  # Mutation scale

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]  # Domination sets
            n_dominates = np.zeros(n, dtype=int)  # Number of solutions dominated
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a list of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            distances = np.zeros(len(indices))
            n_obj = F.shape[1]

            for m in range(n_obj):
                obj_vals = F[indices, m]
                sorted_idx = np.argsort(obj_vals)
                sorted_obj = obj_vals[sorted_idx]

                if sorted_obj[0] == sorted_obj[-1]:
                    continue

                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf

                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (
                        (sorted_obj[i + 1] - sorted_obj[i - 1]) /
                        (sorted_obj[-1] - sorted_obj[0] + 1e-30)
                    )

            return distances

        def tournament_select(pop_X, pop_F, fronts, k=2):
            """Binary tournament selection based on non-dominated rank and crowding distance."""
            # Compute crowding distance per front
            front_crowding = {}
            for front in fronts:
                if len(front) > 0:
                    front_crowding[front] = crowding_distance(pop_F, front)

            def compare(i, j):
                # Find front ranks
                rank_i = next((fi for fi, front in enumerate(fronts) if i in front), 0)
                rank_j = next((fi for fi, front in enumerate(fronts) if j in front), 0)

                if rank_i != rank_j:
                    return rank_i < rank_j

                # Same front, use crowding distance
                front_i = fronts[rank_i]
                front_j = fronts[rank_j]
                cd_i = front_crowding[front_i][np.where(np.array(front_i) == i)[0][0]]
                cd_j = front_crowding[front_j][np.where(np.array(front_j) == j)[0][0]]
                return cd_i > cd_j

            # Tournament
            idx1 = rng.integers(len(pop_X))
            idx2 = rng.integers(len(pop_X))
            if compare(idx1, idx2):
                return idx1
            else:
                return idx2

        def crossover(x1, x2):
            """Simulated binary crossover (SBX)."""
            if rng.random() > crossover_rate:
                return x1.copy(), x2.copy()

            eta = 20
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)

            for d in range(dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (2 * (1 - u)) ** (1 / (eta + 1))

                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])

            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(x):
            """Polynomial mutation."""
            if rng.random() > mutation_rate:
                return x.copy()

            x_new = x.copy()
            eta = 20
            for d in range(dim):
                if rng.random() < 0.5:
                    continue
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                range_d = hi[d] - lo[d]
                x_new[d] = x[d] + delta * range_d
                x_new[d] = np.clip(x_new[d], lo[d], hi[d])

            return x_new

        # Main evolution loop
        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                # Check time
                if time.time() > deadline - 5:
                    break

                # Select two parents
                p1 = tournament_select(X, F, fronts)
                p2 = tournament_select(X, F, fronts)

                # Crossover
                c1, c2 = crossover(X[p1], X[p2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)

                if used < budget and time.time() < deadline - 5:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = len(combined_X)

            # Non-dominated sorting on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
