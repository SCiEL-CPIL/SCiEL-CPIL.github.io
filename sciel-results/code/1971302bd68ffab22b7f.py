"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, keep safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        # For 200 budget, pop=20 gives 10 generations
        pop_size = max(10, min(40, budget // 5))
        # Ensure we can fit at least one generation
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 4)

        rng = np.random.default_rng()

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is a list of indices."""
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

            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(current_front)

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

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_front = F[front_indices]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F_front[sorted_idx[k + 1], m] - F_front[sorted_idx[k - 1], m]) / f_range

            return dist

        def select_parents(F, X, n_select):
            """Select parents using tournament selection based on rank and crowding."""
            fronts = fast_non_dominated_sort(F)
            n = len(F)
            rank = np.zeros(n, dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Compute crowding distances per front
            cd = np.zeros(n)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(F, front)

            # Tournament selection
            selected = []
            for _ in range(n_select):
                candidates = rng.integers(n, size=2)
                i, j = candidates
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    if cd[i] > cd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def crossover(parent1, parent2):
            """SBX crossover."""
            if rng.random() < 0.9:
                eta = 15.0
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1 / (eta + 1))
                    child1[d] = 0.5 * ((1 + beta) * parent1[d] + (1 - beta) * parent2[d])
                    child2[d] = 0.5 * ((1 - beta) * parent1[d] + (1 + beta) * parent2[d])
                return child1, child2
            else:
                return parent1.copy(), parent2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20.0
            x_mut = x.copy()
            for d in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[d] += delta_q * (hi[d] - lo[d])
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Select parents
            n_parents = pop_size
            parent_indices = select_parents(F, X, n_parents)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for i in range(0, n_parents - 1, 2):
                if used + len(offspring_X) >= budget:
                    break
                p1 = X[parent_indices[i]]
                p2 = X[parent_indices[(i + 1) % n_parents]]
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
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

            # Combine populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_n = len(combined_X)

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
                    # Need to sort by crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(zip(cd, front), key=lambda x: -x[0])
                    for k in range(remaining):
                        idx = sorted_front[k][1]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END
