"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(100, max(50, budget // 2))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Crossover and mutation parameters
        cr = 0.9
        F_mut = 0.5

        def fast_non_dominated_sort(F):
            """Compute non-dominated fronts."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            for i in range(n):
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
                fronts.append(next_front)

            return fronts[:-1]  # Remove the last empty front

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                sorted_idx = np.argsort(obj_vals)
                obj_sorted = obj_vals[sorted_idx]
                if obj_sorted[-1] - obj_sorted[0] < 1e-12:
                    dist[sorted_idx] = np.inf
                else:
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (obj_sorted[i + 1] - obj_sorted[i - 1]) / (obj_sorted[-1] - obj_sorted[0])
            return dist

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Evaluate population fitness
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.full(pop_size, -1, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Compute crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(F, front)

            # Selection: binary tournament
            def tournament():
                a, b = rng.integers(pop_size), rng.integers(pop_size)
                if rank[a] < rank[b]:
                    return a
                elif rank[b] < rank[a]:
                    return b
                else:
                    return a if cd[a] > cd[b] else b

            # Create offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)

            for i in range(pop_size):
                # DE mutation
                if rng.random() < 0.8:
                    # Differential evolution
                    p1, p2, p3 = rng.choice(pop_size, size=3, replace=False)
                    v = X[p1] + F_mut * (X[p2] - X[p3])
                else:
                    # Random mutation
                    v = X[rng.integers(pop_size)] + 0.1 * (hi - lo) * rng.standard_normal(dim)

                # Crossover
                cross_mask = rng.random(dim) < cr
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True
                child = np.where(cross_mask, v, X[tournament()])
                child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

                # Check budget and time
                if used >= budget or time.time() > deadline:
                    break

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X[:used - pop_size]]) if used > pop_size else X
            F_all = np.vstack([F, offspring_F[:used - pop_size]]) if used > pop_size else F
            n_all = len(X_all)

            # Sort by rank, then crowding distance
            fronts_all = fast_non_dominated_sort(F_all)
            rank_all = np.full(n_all, -1, dtype=int)
            for i, front in enumerate(fronts_all):
                for idx in front:
                    rank_all[idx] = i

            cd_all = np.zeros(n_all)
            for front in fronts_all:
                if len(front) > 0:
                    cd_all[front] = crowding_distance(F_all, front)

            # Select next population
            if n_all > pop_size:
                # Sort by (rank, -cd)
                sort_keys = np.lexsort((-cd_all, rank_all))
                X = X_all[sort_keys[:pop_size]]
                F = F_all[sort_keys[:pop_size]]
            else:
                X = X_all
                F = F_all

        return X
# EVOLVE-BLOCK-END
