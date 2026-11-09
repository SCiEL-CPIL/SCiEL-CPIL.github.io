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
        start_time = time.time()
        deadline = 300.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance diversity and budget usage
        pop_size = 50
        if self.budget < 100:
            pop_size = max(10, self.budget // 4)

        # Ensure pop_size doesn't exceed budget
        pop_size = min(pop_size, self.budget)

        # Hyperparameters
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability
        p_m = 1.0 / dim  # Mutation probability per gene

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sort
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].add(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        # Helper: crowding distance
        def crowding_distance(F_arr, front_indices):
            if len(front_indices) == 0:
                return {}
            if len(front_indices) == 1:
                return {front_indices[0]: float('inf')}

            dist = {i: 0.0 for i in front_indices}
            n_objs = F_arr.shape[1]

            for m in range(n_objs):
                sorted_front = sorted(front_indices, key=lambda i: F_arr[i, m])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                f_min = F_arr[sorted_front[0], m]
                f_max = F_arr[sorted_front[-1], m]
                if f_max == f_min:
                    continue

                for idx in range(1, len(sorted_front) - 1):
                    i = sorted_front[idx]
                    prev_i = sorted_front[idx - 1]
                    next_i = sorted_front[idx + 1]
                    dist[i] += (F_arr[next_i, m] - F_arr[prev_i, m]) / (f_max - f_min)

            return dist

        # Helper: SBX crossover
        def sbx_crossover(x1, x2):
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)

            if np.random.rand() < p_c:
                for i in range(dim):
                    if np.random.rand() < 0.5:
                        if x1[i] == x2[i]:
                            c1[i] = x1[i]
                            c2[i] = x2[i]
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                            c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                            c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    else:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
            else:
                c1 = x1.copy()
                c2 = x2.copy()

            return c1, c2

        # Helper: polynomial mutation
        def polynomial_mutation(x):
            x_mut = x.copy()
            for i in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    xmut1 = x[i] + delta1 * (hi[i] - lo[i])
                    xmut2 = x[i] + delta2 * (hi[i] - lo[i])
                    if np.random.rand() < 0.5:
                        x_mut[i] = np.clip(xmut1, lo[i], hi[i])
                    else:
                        x_mut[i] = np.clip(xmut2, lo[i], hi[i])
            return x_mut

        # Helper: tournament selection
        def tournament_select(F_arr, ranks, crowding, size=2):
            candidates = np.random.randint(0, len(F_arr), size)
            best = None
            best_rank = float('inf')
            best_crowd = -1.0
            for c in candidates:
                r = ranks[c]
                cr = crowding[c]
                if r < best_rank or (r == best_rank and cr > best_crowd):
                    best = c
                    best_rank = r
                    best_crowd = cr
            return best

        # Main evolution loop
        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Compute ranks and crowding
            fronts, ranks = fast_non_dominated_sort(F)
            crowding = {i: 0.0 for i in range(len(F))}
            for front in fronts:
                cdist = crowding_distance(F, front)
                for idx, val in cdist.items():
                    crowding[idx] = val

            # Generate offspring
            n_crossovers = pop_size // 2
            if pop_size % 2 == 1:
                n_crossovers += 1

            offspring_X = []
            offspring_F = []

            for _ in range(n_crossovers):
                idx1 = tournament_select(F, ranks, crowding)
                idx2 = tournament_select(F, ranks, crowding)

                c1, c2 = sbx_crossover(X[idx1], X[idx2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                offspring_X.append(c1)
                offspring_X.append(c2)

            # Evaluate offspring
            for off in offspring_X:
                if used >= self.budget:
                    break
                if time.time() - start_time > deadline:
                    break
                off_F = problem(off)
                offspring_F.append(off_F)
                used += 1

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X[:len(offspring_F)])])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort on combined
            c_fronts, c_ranks = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            next_ranks = [0] * len(combined_F)
            next_crowding = {i: 0.0 for i in range(len(combined_F))}

            for front in c_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks[idx] = c_ranks[idx]
                else:
                    # Need to truncate this front
                    cdist = crowding_distance(combined_F, front)
                    for idx in front:
                        next_crowding[idx] = cdist.get(idx, 0.0)
                    remaining = pop_size - len(next_X)
                    sorted_front = sorted(front, key=lambda i: next_crowding[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks[idx] = c_ranks[idx]
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
