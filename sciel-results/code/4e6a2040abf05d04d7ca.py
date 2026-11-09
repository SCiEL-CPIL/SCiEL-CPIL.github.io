import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size and generation budget
        pop_size = 60
        # We have budget evaluations total. Initial pop uses pop_size.
        # Each generation: pop_size children evaluated.
        max_gens = max(1, (self.budget - pop_size) // pop_size)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts, each a list of indices."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        domination_count[i] += 1
                if domination_count[i] == 0:
                    fronts.append([i])

            k = 0
            while fronts[k]:
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F_arr[indices, m]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                min_v = vals[sorted_idx[0]]
                max_v = vals[sorted_idx[-1]]
                if max_v - min_v < 1e-15:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_v - min_v)
            return cd

        def sbx_crossover(x1, x2, eta=15.0):
            """Simulated Binary Crossover."""
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = child2[i] = x1[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = np.clip(child1[i], lo[i], hi[i])
                        child2[i] = np.clip(child2[i], lo[i], hi[i])
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            return child1, child2

        def polynomial_mutate(x, eta=20.0):
            """Polynomial mutation."""
            xm = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    delta1 = (1 + 2 * (x[i] - lo[i]) / span[i]) ** (eta + 1) if span[i] > 1e-14 else 2.0
                    delta2 = (1 + 2 * (hi[i] - x[i]) / span[i]) ** (eta + 1) if span[i] > 1e-14 else 2.0
                    if u < 0.5:
                        deltaq = (1 + 2.0 * (rng.random() - u)) ** (1.0 / (eta + 1)) - 1
                    else:
                        deltaq = 1 - (1 + 2.0 * (u - 0.5)) ** (1.0 / (eta + 1))
                    xm[i] += deltaq * min(delta1, delta2) * span[i]
                    xm[i] = np.clip(xm[i], lo[i], hi[i])
            return xm

        # Main evolutionary loop
        for gen in range(max_gens):
            if used >= self.budget:
                break
            if time.time() > deadline - 30:
                break

            # Evaluate fitness is already done for current population
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(F, np.array(front))
                else:
                    cd[front] = np.inf

            # Selection: tournament on (rank, crowding distance)
            def tournament_select(pop_size_sel):
                selected = []
                for _ in range(pop_size_sel):
                    i1 = rng.integers(pop_size)
                    i2 = rng.integers(pop_size)
                    if rank[i1] < rank[i2]:
                        selected.append(i1)
                    elif rank[i2] < rank[i1]:
                        selected.append(i2)
                    else:
                        if cd[i1] > cd[i2]:
                            selected.append(i1)
                        else:
                            selected.append(i2)
                return selected

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)
            if n_offspring <= 0:
                break

            # Pair up parents
            parents = tournament_select(2 * n_offspring)
            for k in range(0, 2 * n_offspring - 1, 2):
                if time.time() > deadline - 20:
                    break
                c1, c2 = sbx_crossover(X[parents[k]], X[parents[k + 1]])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= self.budget:
                    break
                if k + 2 < 2 * n_offspring:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:n_offspring])
            offspring_F = np.array(offspring_F[:n_offspring])

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)

            # Non-dominated sort on combined
            fronts_c = fast_non_dominated_sort(combined_F)

            # Assign ranks
            rank_c = np.zeros(n_combined, dtype=int)
            for r, front in enumerate(fronts_c):
                for idx in front:
                    rank_c[idx] = r

            # Crowding distance
            cd_c = np.zeros(n_combined)
            for front in fronts_c:
                if len(front) > 2:
                    cd_c[front] = crowding_distance(combined_F, np.array(front))
                else:
                    cd_c[front] = np.inf

            # Select best pop_size individuals
            selected_indices = []
            for front in fronts_c:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = pop_size - len(selected_indices)
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd_c[i], reverse=True)
                    selected_indices.extend(front_sorted[:remaining])
                    break

            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

        # Return the archive (all evaluated points that are non-dominated)
        fronts = fast_non_dominated_sort(F)
        archive_indices = []
        for front in fronts:
            archive_indices.extend(front)

        return X[archive_indices]
