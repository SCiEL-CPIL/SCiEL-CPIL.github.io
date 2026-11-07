"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(50, max(10, n // 4))

        rng = np.random.default_rng(42)

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sort
        def non_dominated_sort(F):
            """Returns list of fronts, each a list of indices."""
            m = len(F)
            S = [[] for _ in range(m)]  # S[i]: indices dominated by i
            n_dom = [0] * m  # number of points dominated by i
            fronts = []

            for i in range(m):
                for j in range(i + 1, m):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[j] += 1

            current_front = [i for i in range(m) if n_dom[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            """Assign crowding distances to points in indices."""
            m = len(indices)
            if m <= 2:
                return [float('inf')] * m
            cd = [0.0] * m
            for k in range(n_obj):
                vals = np.array([F[i][k] for i in indices])
                order = np.argsort(vals)
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range == 0:
                    continue
                for i in range(1, m - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return cd

        # Helper: binary tournament selection
        def binary_tournament(F, fronts, cd_map):
            """Select two parents using non-dominated rank then crowding distance."""
            def rank_of(idx):
                for r, front in enumerate(fronts):
                    if idx in front:
                        return r
                return len(fronts)

            i = rng.integers(len(F))
            j = rng.integers(len(F))
            ri, rj = rank_of(i), rank_of(j)
            if ri < rj:
                return i
            elif rj < ri:
                return j
            # Same rank, use crowding distance
            ci = cd_map.get(i, 0)
            cj = cd_map.get(j, 0)
            return i if ci >= cj else j

        # Main loop
        while used < n:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Compute crowding distances per front
            cd_map = {}
            for front in fronts:
                cds = crowding_distance(F, front)
                for idx, cd in zip(front, cds):
                    cd_map[idx] = cd

            # Generate offspring
            offspring = []
            while len(offspring) < pop_size and used < n:
                if time.time() > deadline:
                    break
                p1 = binary_tournament(F, fronts, cd_map)
                p2 = binary_tournament(F, fronts, cd_map)

                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 15)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 15)
                        child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                if rng.random() < 0.9:
                    eta = 20
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                            child[d] += delta_q * (hi[d] - lo[d])
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring.append((child, fc))

            if not offspring:
                break

            # Combine parents and offspring
            X_all = np.vstack([X, np.array([o[0] for o in offspring])])
            F_all = np.array([F, np.array([o[1] for o in offspring])])

            # Non-dominated sort on combined
            fronts_all = non_dominated_sort(F_all)

            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    remaining = pop_size - len(X_new)
                    cds = crowding_distance(F_all, front)
                    order = np.argsort(cds)[::-1]  # sort by CD descending
                    for pos in order[:remaining]:
                        X_new.append(X_all[front[pos]])
                        F_new.append(F_all[front[pos]])
                    break

            if X_new:
                X = np.array(X_new)
                F = np.array(F_new)

        return X
# EVOLVE-BLOCK-END
