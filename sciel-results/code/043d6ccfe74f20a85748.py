"""NSGA-II style multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(100, max(20, budget // 3))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Initialize population uniformly
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(fitnesses):
            """Return list of fronts (indices), where each front is a list of indices."""
            n = len(fitnesses)
            S = [[] for _ in range(n)]  # set of solutions dominated by i
            n_dominates = [0] * n       # number of solutions dominated by i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        n_dominates[i] += 1

            current_front = []
            for i in range(n):
                if n_dominates[i] == 0:
                    current_front.append(i)

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1

            return fronts

        def crowding_distance(fitnesses, front_indices):
            """Compute crowding distance for solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            distances = [0.0] * n
            for m in range(n_obj):
                sorted_indices = sorted(front_indices, key=lambda idx: fitnesses[idx][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                f_min = fitnesses[sorted_indices[0]][m]
                f_max = fitnesses[sorted_indices[-1]][m]
                if f_max - f_min == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (fitnesses[sorted_indices[i + 1]][m] - fitnesses[sorted_indices[i - 1]][m]) / (f_max - f_min)

            return distances

        def tournament_select(fronts, fitnesses, X, pop_size):
            """Select pop_size individuals using binary tournament on rank + crowding distance."""
            # Assign ranks
            ranks = [0] * len(fitnesses)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances for each front
            crowding = [0.0] * len(fitnesses)
            for front in fronts:
                cd = crowding_distance(fitnesses, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]

            selected_X = []
            selected_F = []
            for _ in range(pop_size):
                # Binary tournament
                i1 = np.random.randint(len(fitnesses))
                i2 = np.random.randint(len(fitnesses))
                if ranks[i1] < ranks[i2]:
                    winner = i1
                elif ranks[i2] < ranks[i1]:
                    winner = i2
                else:
                    if crowding[i1] > crowding[i2]:
                        winner = i1
                    else:
                        winner = i2
                selected_X.append(X[winner].copy())
                selected_F.append(F[winner].copy())

            return np.array(selected_X), np.array(selected_F)

        def crossover(x1, x2):
            """Simulated binary crossover (SBX)."""
            if np.random.random() < 0.9:
                eta = 20
                child1 = x1.copy()
                child2 = x2.copy()
                for j in range(dim):
                    if x1[j] != x2[j]:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        child1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        child2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutation(x):
            """Polynomial mutation."""
            eta = 20
            for j in range(dim):
                if np.random.random() < 0.8:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x[j] += delta * (hi[j] - lo[j])
            return np.clip(x, lo, hi)

        while used < budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            offspring_budget = min(pop_size, budget - used)

            for _ in range(offspring_budget):
                # Select parents
                parent1_idx = np.random.randint(pop_size)
                parent2_idx = np.random.randint(pop_size)
                while parent2_idx == parent1_idx:
                    parent2_idx = np.random.randint(pop_size)

                c1, c2 = crossover(X[parent1_idx], X[parent2_idx])
                c1 = mutation(c1)
                c2 = mutation(c2)

                f1 = problem(c1)
                f2 = problem(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                used += 2

                if used >= budget:
                    break

            offspring_X = np.array(offspring_X[:offspring_budget])
            offspring_F = np.array(offspring_F[:offspring_budget])

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_indices = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        return X
# EVOLVE-BLOCK-END
