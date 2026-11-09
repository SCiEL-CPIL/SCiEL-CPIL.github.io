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
        deadline = start_time + 295.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(60, max(20, budget // 4))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_nondominated_sort(F_arr):
            """Return list of fronts (each a list of indices)."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated = [list() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[j].append(i)
                        domination_count[i] += 1
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            distances = np.zeros(n)
            for m in range(n_obj):
                f_vals = F_arr[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(f_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (f_vals[sorted_idx[k + 1]] - f_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        def tournament_select(F_arr, fronts, crowding_dists, size=2):
            """Select one individual via tournament."""
            idx_a = rng.integers(len(F_arr))
            idx_b = rng.integers(len(F_arr))
            front_a = fronts.index(next(f for f in fronts if idx_a in f)) if idx_a in [x for f in fronts for x in f] else 0
            front_b = fronts.index(next(f for f in fronts if idx_b in f)) if idx_b in [x for f in fronts for x in f] else 0

            # Find which front each index belongs to
            front_rank_a = 0
            for fi, f in enumerate(fronts):
                if idx_a in f:
                    front_rank_a = fi
                    break
            front_rank_b = 0
            for fi, f in enumerate(fronts):
                if idx_b in f:
                    front_rank_b = fi
                    break

            if front_rank_a < front_rank_b:
                return idx_a
            elif front_rank_b < front_rank_a:
                return idx_b
            # Same front, use crowding distance
            cd_a = crowding_dists[front_rank_a][idx_a] if idx_a in [x for x in fronts[front_rank_a]] else 0
            cd_b = crowding_dists[front_rank_b][idx_b] if idx_b in [x for x in fronts[front_rank_b]] else 0
            return idx_a if cd_a > cd_b else idx_b

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            child1 = x1.copy()
            child2 = x2.copy()
            for j in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[j] - x2[j]) > 1e-14:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        c1 = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        c2 = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                        child1[j] = c1
                        child2[j] = c2
            return child1, child2

        def mutate(x):
            """Polynomial mutation."""
            eta = 20.0
            child = x.copy()
            for j in range(dim):
                if rng.random() < 0.5:
                    span = hi[j] - lo[j]
                    if span < 1e-14:
                        continue
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[j] = x[j] + delta * span
                    child[j] = np.clip(child[j], lo[j], hi[j])
            return child

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Sort into non-dominated fronts
            fronts = fast_nondominated_sort(F)

            # Compute crowding distances for each front
            crowding_dists = []
            for front in fronts:
                cd = crowding_distance(F, front)
                crowding_dists.append(cd)

            # Create offspring
            offspring = []
            n_offspring = pop_size
            while len(offspring) < n_offspring:
                if time.time() > deadline:
                    break

                # Select two parents
                p1 = tournament_select(F, fronts, crowding_dists)
                p2 = tournament_select(F, fronts, crowding_dists)

                # Crossover
                c1, c2 = crossover(X[p1], X[p2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = problem(c1)
                used += 1
                if used > budget:
                    break
                offspring.append((c1, f1))

                if used >= budget:
                    break

                if time.time() > deadline:
                    break

                f2 = problem(c2)
                used += 1
                if used > budget:
                    break
                offspring.append((c2, f2))

            if not offspring:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array([o[0] for o in offspring])])
            F_combined = np.vstack([F, np.array([o[1] for o in offspring])])

            # Sort combined into fronts
            combined_fronts = fast_nondominated_sort(F_combined)

            # Select next generation
            X_new = []
            F_new = []
            for front in combined_fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_combined, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break

            X = np.array(X_new) if X_new else X
            F = np.array(F_new) if F_new else F

        return X

# EVOLVE-BLOCK-END
