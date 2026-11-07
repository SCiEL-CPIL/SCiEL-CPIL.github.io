"""NSGA-II multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 280.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size: use a reasonable fraction of budget
        pop_size = min(30, budget // 6)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        evals = 0

        for i in range(pop_size):
            F[i] = problem(X[i])
            evals += 1

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (each is a list of indices)."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.ones(n) * np.inf

            cd = np.zeros(n)
            idx = np.array(indices)
            F_sub = F_arr[idx]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / (f_max - f_min)

            return cd

        def tournament_select(F_arr, fronts, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            rank = np.zeros(len(F_arr), dtype=int)
            for front_id, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_id

            cd_arr = np.zeros(len(F_arr))
            for front_id, front in enumerate(fronts):
                cd_front = crowding_distance(F_arr, front)
                for pos, idx in enumerate(front):
                    cd_arr[idx] = cd_front[pos]

            candidates = rng.integers(0, len(F_arr), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd_arr[c] > cd_arr[best]):
                    best = c
            return best

        # Main NSGA-II loop
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))

            fronts = fast_non_dominated_sort(F)

            for i in range(pop_size):
                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)

                # Blend crossover
                if rng.random() < 0.9:
                    alpha = rng.random()
                    child = (1 - alpha) * X[p1] + alpha * X[p2]
                else:
                    child = X[p1].copy()

                # Mutation
                for j in range(dim):
                    if rng.random() < 0.3:
                        child[j] += 0.1 * span[j] * rng.standard_normal()

                child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals += 1

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_n = 2 * pop_size

            combined_fronts = fast_non_dominated_sort(combined_F)

            new_X = np.empty((pop_size, dim))
            new_F = np.empty((pop_size, n_obj))
            count = 0

            for front in combined_fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    remaining = pop_size - count
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
