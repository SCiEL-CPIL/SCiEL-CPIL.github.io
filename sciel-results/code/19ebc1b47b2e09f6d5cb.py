"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 290.0  # safety margin before 300s kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng()  # uses np.random state set by evaluator

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        pop_size = 40

        # Stratified initialization
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                X[i, j] = lo[j] + (hi[j] - lo[j]) * ((i * dim + j + 0.5) / (pop_size * dim))
        # Add some random noise
        X += 0.02 * (hi - lo) * rng.standard_normal((pop_size, dim))
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
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
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F_sub[order[-1], k] - F_sub[order[0], k] > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (
                            F_sub[order[-1], k] - F_sub[order[0], k]
                        )
            return cd

        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            child = x1.copy()
            mask = rng.random(dim) < 0.5
            for j in np.where(mask)[0]:
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
            return np.clip(child, lo, hi)

        def polynomial_mutate(x, lo, hi, eta=20.0, prob=1.0 / dim):
            x = x.copy()
            mask = rng.random(dim) < prob
            for j in np.where(mask)[0]:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x[j] += delta * (hi[j] - lo[j])
            return np.clip(x, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Tournament selection
            def select_tournament():
                i, j = rng.integers(pop_size, size=2)
                # Determine rank
                rank_i, rank_j = 0, 0
                for r, front in enumerate(fronts):
                    if i in front:
                        rank_i = r
                    if j in front:
                        rank_j = r
                    if rank_i > 0 and rank_j > 0:
                        break
                if rank_i != rank_j:
                    return i if rank_i < rank_j else j
                # Crowding distance tie-break
                cd = crowding_distance(F, list(range(pop_size)))
                return i if cd[i] > cd[j] else j

            parents1 = [select_tournament() for _ in range(pop_size // 2)]
            parents2 = [select_tournament() for _ in range(pop_size // 2)]

            offspring_X = []
            for i in range(pop_size // 2):
                p1, p2 = parents1[i], parents2[i]
                child = sbx_crossover(X[p1], X[p2], lo, hi)
                child = polynomial_mutate(child, lo, hi)
                offspring_X.append(child)

            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += len(offspring_X)

            # Combine and select next population
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, offspring_F])

            fronts_comb = fast_non_dominated_sort(F_combined)
            next_idx = []
            for front in fronts_comb:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(-cd)
                    remaining = pop_size - len(next_idx)
                    next_idx.extend([front[i] for i in order[:remaining]])
                    break

            X = X_combined[next_idx]
            F = F_combined[next_idx]

        return X
# EVOLVE-BLOCK-END
