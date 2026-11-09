"""Improved multi-objective optimizer for ZDT4 using NSGA-II with crowding distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Random initialization
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
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
                fronts.append(next_front)
                if not fronts[k]:
                    break

            return fronts[:-1] if fronts[-1] == [] else fronts

        # Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            F_sub = F[indices]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                if F_sub[sorted_idx, m].max() - F_sub[sorted_idx, m].min() == 0:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (
                        F_sub[sorted_idx, m].max() - F_sub[sorted_idx, m].min()
                    )

            return cd

        # Selection: binary tournament based on rank and crowding
        def tournament_select(pop_X, pop_F, fronts, cd_map):
            a, b = rng.integers(0, len(pop_X), size=2)
            rank_a = next(i for i, f in enumerate(fronts) if a in f)
            rank_b = next(i for i, f in enumerate(fronts) if b in f)
            if rank_a < rank_b:
                return pop_X[a].copy()
            elif rank_b < rank_a:
                return pop_X[b].copy()
            else:
                cd_a = cd_map.get(a, 0)
                cd_b = cd_map.get(b, 0)
                return pop_X[a].copy() if cd_a >= cd_b else pop_X[b].copy()

        # DE mutation
        def de_mutate(parent, X, F):
            if len(X) < 3:
                return parent + 0.1 * (hi - lo) * rng.standard_normal(dim)
            i1, i2 = rng.choice(len(X), size=2, replace=False)
            CR = 0.8
            F_mut = 0.8
            v = X[i1] + F_mut * (X[i2] - parent)
            # Binary mask
            mask = rng.random(dim) < CR
            # Ensure at least one gene changes
            if not np.any(mask):
                mask[rng.integers(dim)] = True
            child = np.where(mask, v, parent)
            return np.clip(child, lo, hi)

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute ranks
            fronts = fast_non_dominated_sort(F)

            # Compute crowding distance for each front
            cd_map = {}
            for front in fronts:
                if len(front) > 2:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_map[idx] = cd[i]
                else:
                    for idx in front:
                        cd_map[idx] = np.inf

            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            offspring_X = []
            offspring_F = []
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                parent = tournament_select(X, F, fronts, cd_map)
                child = de_mutate(parent, X, F)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)

            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
