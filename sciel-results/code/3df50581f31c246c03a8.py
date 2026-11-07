"""Multi-objective optimizer on ZDT3: NSGA-II style with crowding distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style evolutionary multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(budget, max(50, budget // 2))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Use a fixed seed for reproducibility within the call
        rng = np.random.default_rng(42)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Record all evaluated points (for scoring, all recorded vectors count)
        all_F = [F.copy()]

        def fast_non_dominated_sort(F_mat):
            """Return list of fronts, each a list of indices."""
            n = len(F_mat)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
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

        def crowding_distance(F_mat, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F_mat[front_indices, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-15:
                    distances[:] = np.inf
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        def tournament_select(fronts, F_mat, n_parents):
            """Select n_parents individuals using tournament on fronts then crowding."""
            # Assign front rank
            front_rank = np.zeros(len(F_mat), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    front_rank[idx] = rank

            selected = []
            for _ in range(n_parents):
                # Tournament of size 2
                a = rng.integers(len(F_mat))
                b = rng.integers(len(F_mat))
                if front_rank[a] < front_rank[b]:
                    selected.append(a)
                elif front_rank[b] < front_rank[a]:
                    selected.append(b)
                else:
                    # Same front: use crowding distance
                    front = fronts[front_rank[a]]
                    cd = crowding_distance(F_mat, front)
                    pos_a = np.where(np.array(front) == a)[0][0]
                    pos_b = np.where(np.array(front) == b)[0][0]
                    if cd[pos_a] >= cd[pos_b]:
                        selected.append(a)
                    else:
                        selected.append(b)
            return selected

        # Crossover and mutation
        def crossover(x1, x2):
            """SBX crossover."""
            if rng.random() < 0.9:  # crossover probability
                eta = 20.0
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x, mutation_rate=0.1, sigma=0.1):
            """Polynomial mutation."""
            x_mut = x.copy()
            mask = rng.random(dim) < mutation_rate
            if np.any(mask):
                eta_m = 20.0
                for i in np.where(mask)[0]:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    x_mut[i] += delta * (hi[i] - lo[i]) * sigma
            return np.clip(x_mut, lo, hi)

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline - 5:
                break

            # Evaluate current population
            fronts = fast_non_dominated_sort(F)

            # Create offspring population of same size
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            while len(offspring_X) < n_offspring:
                # Select two parents
                parents = tournament_select(fronts, F, 2)
                p1, p2 = X[parents[0]], X[parents[1]]

                # Crossover
                c1, c2 = crossover(p1, p2)

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                fc1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(fc1)

                if len(offspring_X) < n_offspring:
                    fc2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(fc2)

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Record offspring for scoring
            all_F.append(np.array(offspring_F))

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: -cd[i])
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Return the best non-dominated points
        return X

# EVOLVE-BLOCK-END
