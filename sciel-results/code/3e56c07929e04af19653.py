"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 40

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        evals = N

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort, returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                current_front = next_front
                if current_front:
                    fronts.append(current_front)

            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            distances = [0.0] * len(indices)
            n = len(indices)
            for m in range(n_obj):
                obj_vals = F[indices, m]
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 1e-12:
                    for k in range(1, n - 1):
                        distances[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / obj_range
            return distances

        while evals < budget:
            if time.time() > deadline:
                break

            offspring_X = []
            offspring_F = []

            while len(offspring_X) < N and evals < budget:
                # Tournament selection based on non-dominated rank + crowding
                fronts = fast_non_dominated_sort(F)
                n = len(F)
                rank = [0] * n
                for front_idx, front in enumerate(fronts):
                    for i in front:
                        rank[i] = front_idx

                def select_one():
                    idx1 = rng.integers(n)
                    idx2 = rng.integers(n)
                    if rank[idx1] < rank[idx2]:
                        return int(idx1)
                    elif rank[idx2] < rank[idx1]:
                        return int(idx2)
                    else:
                        front_idx = rank[idx1]
                        front = fronts[front_idx]
                        cd = crowding_distance(F, front)
                        pos1 = front.index(int(idx1))
                        pos2 = front.index(int(idx2))
                        if cd[pos1] > cd[pos2]:
                            return int(idx1)
                        else:
                            return int(idx2)

                p1 = select_one()
                p2 = select_one()

                # SBX crossover
                alpha = 20
                x1 = X[p1]
                x2 = X[p2]
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5 and abs(x1[d] - x2[d]) > 1e-12:
                        u = rng.random()
                        if u <= 1.0 / (alpha + 1):
                            beta = (2 * u) ** (1.0 / (alpha + 1))
                        elif u >= 1.0 - 1.0 / (alpha + 1):
                            beta = (2 * (1 - u)) ** (1.0 / (alpha + 1))
                        else:
                            beta = 1.0
                        child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                    else:
                        if rng.random() < 0.5:
                            child1[d] = x1[d]
                            child2[d] = x2[d]
                        else:
                            child1[d] = x2[d]
                            child2[d] = x1[d]

                # Polynomial mutation
                eta = 20
                for d in range(dim):
                    if rng.random() < 0.1:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child1[d] += delta * (hi[d] - lo[d])
                    if rng.random() < 0.1:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child2[d] += delta * (hi[d] - lo[d])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                evals += 1

                if evals < budget:
                    offspring_X.append(child2)
                    offspring_F.append(problem(child2))
                    evals += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])

            fronts = fast_non_dominated_sort(all_F)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for i in front:
                        new_X.append(all_X[i])
                        new_F.append(all_F[i])
                else:
                    remaining = N - len(new_X)
                    cd = crowding_distance(all_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                    for k in range(min(remaining, len(front))):
                        idx = front[sorted_front[k]]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
