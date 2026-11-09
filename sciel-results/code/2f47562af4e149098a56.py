# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(12345)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size: keep it moderate so each generation is fast
        pop_size = 40

        # Initial population (uniform random)
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
            """Compute crowding distance for indices in a front (returns array aligned with front_indices)."""
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

        def tournament_select(F_arr, front_ranks, crowding_map, size=2):
            """Select one individual via tournament using front rank then crowding distance."""
            best = -1
            best_rank = None
            best_cd = None
            for _ in range(size):
                idx = int(rng.integers(len(F_arr)))
                r = front_ranks[idx]
                cd = crowding_map.get(idx, 0.0)
                if best_rank is None or r < best_rank or (r == best_rank and cd > best_cd):
                    best = idx
                    best_rank = r
                    best_cd = cd
            return best

        def sbx_crossover(x1, x2):
            """Simulated binary crossover (vectorized)."""
            eta = 15.0
            mask = rng.random(dim) < 0.5
            child1 = x1.copy()
            child2 = x2.copy()
            diff = np.abs(x1 - x2)
            valid = diff > 1e-14
            m = mask & valid
            if np.any(m):
                u = rng.random(np.count_nonzero(m))
                beta = np.where(
                    u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                )
                c1 = 0.5 * ((1.0 + beta) * x1[m] + (1.0 - beta) * x2[m])
                c2 = 0.5 * ((1.0 - beta) * x1[m] + (1.0 + beta) * x2[m])
                child1[m] = c1
                child2[m] = c2
            return child1, child2

        def polynomial_mutate(x):
            """Polynomial mutation (vectorized)."""
            eta = 20.0
            child = x.copy()
            mask = rng.random(dim) < 0.5
            if not np.any(mask):
                return child
            u = rng.random(np.count_nonzero(mask))
            delta = np.where(
                u < 0.5,
                (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            )
            span = hi - lo
            child[mask] = x[mask] + delta * span[mask]
            return np.clip(child, lo, hi)

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_nondominated_sort(F)

            # Front rank per individual
            front_ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    front_ranks[idx] = rank

            # Crowding distance map (index -> cd)
            crowding_map = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowding_map[idx] = cd[i]

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            while len(offspring_X) < n_offspring and used < budget:
                if time.time() > deadline:
                    break

                p1 = tournament_select(F, front_ranks, crowding_map)
                p2 = tournament_select(F, front_ranks, crowding_map)

                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)

                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)

                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])

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
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return X

# EVOLVE-BLOCK-END
