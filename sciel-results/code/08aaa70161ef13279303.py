"""Multi-objective optimizer for RE37: NSGA-II with crowding distance and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 40

        # Initial population
        np.random.seed(42)
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # All evaluated points so far (for final hypervolume)
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F_mat):
            """Return list of fronts, each a list of indices."""
            n = len(F_mat)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F_mat, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_mat[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)
            return dist

        def binary_tournament(fronts, F_mat, idx_a, idx_b):
            """Return the better index between idx_a and idx_b."""
            front_a = None
            front_b = None
            for i, front in enumerate(fronts):
                if idx_a in front:
                    front_a = i
                    break
            for i, front in enumerate(fronts):
                if idx_b in front:
                    front_b = i
                    break
            if front_a is None or front_b is None:
                return idx_a
            if front_a < front_b:
                return idx_a
            if front_b < front_a:
                return idx_b
            # Same front: use crowding distance
            # Compute crowding for this front
            front_indices = []
            for fi, front in enumerate(fronts):
                if fi == front_a:
                    front_indices = front
                    break
            cd = crowding_distance(F_mat, front_indices)
            cd_map = {}
            for ii, idx in enumerate(front_indices):
                cd_map[idx] = cd[ii]
            if cd_map.get(idx_a, 0) > cd_map.get(idx_b, 0):
                return idx_a
            return idx_b

        def crossover(x1, x2):
            """SBX crossover."""
            eta = 20
            if np.random.random() < 0.5:
                c1 = x1.copy()
                c2 = x2.copy()
            else:
                c1 = x2.copy()
                c2 = x1.copy()
            for j in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[j] - x2[j]) > 1e-14:
                        u = np.random.random()
                        if u <= 1.0 / (eta + 1):
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        elif u >= 1.0 - 1.0 / (eta + 1):
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        else:
                            beta = 1.0
                        c1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        c2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        def mutation(x):
            """Polynomial mutation."""
            eta = 20
            xm = x.copy()
            for j in range(dim):
                if np.random.random() < 0.5:
                    delta = hi[j] - lo[j]
                    if delta < 1e-14:
                        continue
                    u = np.random.random()
                    if u < 0.5:
                        val = (1 + 2 * (x[j] - lo[j]) / delta) ** (1.0 / (eta + 1))
                        xm[j] = x[j] - delta * (1 - val)
                    else:
                        val = (1 + 2 * (hi[j] - x[j]) / delta) ** (1.0 / (eta + 1))
                        xm[j] = x[j] + delta * (1 - val)
            return np.clip(xm, lo, hi)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            for _ in range(N // 2):
                # Tournament selection for parents
                i1 = np.random.randint(N)
                i2 = np.random.randint(N)
                fronts = fast_non_dominated_sort(F)
                parent1 = binary_tournament(fronts, F, i1, i2)

                i3 = np.random.randint(N)
                i4 = np.random.randint(N)
                parent2 = binary_tournament(fronts, F, i3, i4)

                # Crossover
                c1, c2 = crossover(X[parent1], X[parent2])
                # Mutation
                c1 = mutation(c1)
                c2 = mutation(c2)

                offspring_X.append(c1)
                offspring_X.append(c2)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N

            if time.time() > deadline:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            all_X.append(combined_X)
            all_F.append(combined_F)

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # Sort by descending crowding distance
                    for i in range(min(remaining, len(front))):
                        idx = front[order[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Collect all unique non-dominated points from all evaluations
        # Gather all evaluated points
        all_eval_X = []
        all_eval_F = []
        for x_arr, f_arr in zip(all_X, all_F):
            for i in range(len(x_arr)):
                all_eval_X.append(x_arr[i])
                all_eval_F.append(f_arr[i])

        if all_eval_X:
            all_eval_X = np.array(all_eval_X)
            all_eval_F = np.array(all_eval_F)
            # Get non-dominated points
            fronts = fast_non_dominated_sort(all_eval_F)
            if fronts and len(fronts[0]) > 0:
                final_indices = fronts[0]
                final_X = all_eval_X[final_indices]
                final_F = all_eval_F[final_indices]
                return final_X
            else:
                return all_eval_X
        else:
            return X

# EVOLVE-BLOCK-END
