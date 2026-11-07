"""NSGA-II-based multi-objective optimizer for WFG9."""
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

        # Population size
        pop_size = min(60, budget // 3)
        if pop_size < 20:
            pop_size = 20

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        evals = 0
        all_F_list = []

        for i in range(pop_size):
            if evals >= budget:
                break
            F[i] = problem(X[i])
            evals += 1
            all_F_list.append(F[i].copy())

        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            if n == 0:
                return []
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

        def compute_rank_and_cd(F_arr, fronts):
            n = len(F_arr)
            rank = np.zeros(n, dtype=int)
            cd_arr = np.zeros(n)
            for front_id, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_id
                if len(front) > 2:
                    cd_front = crowding_distance(F_arr, front)
                    for pos, idx in enumerate(front):
                        cd_arr[idx] = cd_front[pos]
                else:
                    for idx in front:
                        cd_arr[idx] = np.inf
            return rank, cd_arr

        def tournament_select(rank, cd_arr, k=3):
            n = len(rank)
            candidates = rng.integers(0, n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd_arr[c] > cd_arr[best]):
                    best = c
            return best

        # Main NSGA-II loop
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            fronts = fast_non_dominated_sort(F)
            rank, cd_arr = compute_rank_and_cd(F, fronts)

            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))
            n_offspring = 0

            for i in range(pop_size):
                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                p1 = tournament_select(rank, cd_arr)
                p2 = tournament_select(rank, cd_arr)

                # Blend crossover
                if rng.random() < 0.9:
                    alpha = rng.random()
                    child1 = (1 - alpha) * X[p1] + alpha * X[p2]
                    child2 = (1 - alpha) * X[p2] + alpha * X[p1]
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Mutation
                mutation_rate = 0.3
                for j in range(dim):
                    if rng.random() < mutation_rate:
                        child1[j] += 0.1 * span[j] * rng.standard_normal()
                    if rng.random() < mutation_rate:
                        child2[j] += 0.1 * span[j] * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X[n_offspring] = child1
                offspring_F[n_offspring] = problem(child1)
                evals += 1
                all_F_list.append(offspring_F[n_offspring].copy())
                n_offspring += 1

                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                if n_offspring < pop_size and evals < budget:
                    offspring_X[n_offspring] = child2
                    offspring_F[n_offspring] = problem(child2)
                    evals += 1
                    all_F_list.append(offspring_F[n_offspring].copy())
                    n_offspring += 1

            if n_offspring == 0:
                break

            combined_X = np.vstack([X, offspring_X[:n_offspring]])
            combined_F = np.vstack([F, offspring_F[:n_offspring]])

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
