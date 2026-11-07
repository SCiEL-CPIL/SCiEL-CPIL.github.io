import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_bounds = hi - lo

        # Population size
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)

        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N

        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = F.shape[0]
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                rank += 1
                current_front = next_front
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            n = F.shape[0]
            distances = np.zeros(n)
            if n <= 2:
                return distances

            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    continue

                for m in range(self.n_obj):
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    distances[sorted_idx[0]] = np.inf
                    distances[sorted_idx[-1]] = np.inf

                    obj_min = F[sorted_idx[0], m]
                    obj_max = F[sorted_idx[-1], m]
                    obj_range = obj_max - obj_min
                    if obj_range == 0:
                        continue

                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i - 1]
                        next_idx = sorted_idx[i + 1]
                        distances[idx] += (F[next_idx, m] - F[prev_idx, m]) / obj_range

            return distances

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=15.0):
            x_c = x1.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if x1[i] < x2[i]:
                            beta = 1.0 + (2.0 * (x1[i] - lo[i]) / (x2[i] - x1[i]))
                            alpha = 2.0 - beta ** (-(eta + 1.0))
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                            x_c[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x2[i] - x1[i]))
                        else:
                            beta = 1.0 + (2.0 * (hi[i] - x2[i]) / (x1[i] - x2[i]))
                            alpha = 2.0 - beta ** (-(eta + 1.0))
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                            x_c[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))
            return x_c

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_m = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    delta1 = (x_m[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x_m[i]) / (hi[i] - lo[i])
                    rand = np.random.rand()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                        deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                        deltaq = 1.0 - val ** (1.0 / (eta + 1.0))
                    x_m[i] += deltaq * (hi[i] - lo[i])
                    x_m[i] = np.clip(x_m[i], lo[i], hi[i])
            return x_m

        # Main Loop
        while evals < self.budget:
            if time.time() - start_time > timeout:
                break

            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)

            # Create offspring
            offspring_X = []
            while len(offspring_X) < N:
                # Binary Tournament Selection
                i, j = np.random.choice(N, 2, replace=False)
                if ranks[i] != ranks[j]:
                    p1 = i if ranks[i] < ranks[j] else j
                else:
                    p1 = i if cd[i] > cd[j] else j
                
                i, j = np.random.choice(N, 2, replace=False)
                if ranks[i] != ranks[j]:
                    p2 = i if ranks[i] < ranks[j] else j
                else:
                    p2 = i if cd[i] > cd[j] else j

                # Crossover
                if np.random.rand() < 0.9: # Crossover probability
                    child1 = sbx_crossover(X[p1], X[p2])
                    child2 = sbx_crossover(X[p2], X[p1])
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)

                offspring_X.append(child1)
                if len(offspring_X) < N:
                    offspring_X.append(child2)

            offspring_X = np.array(offspring_X[:N])
            
            # Evaluate offspring
            if time.time() - start_time > timeout:
                break

            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += N

            # Combine and Select
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            ranks_comb = non_dominated_sort(F_comb)
            cd_comb = crowding_distance(F_comb, ranks_comb)
            
            # Selection
            # We want the first N indices from the sorted (rank, -cd)
            # Using argsort on a tuple of arrays
            # Note: np.lexsort sorts by the last key first, so we pass (-cd, rank) to get rank first?
            # No, lexsort((k1, k2)) sorts by k2 then k1.
            # We want primary sort by rank (ascending), secondary by cd (descending).
            # So keys: primary=rank, secondary=-cd.
            # np.lexsort((secondary, primary))
            order = np.lexsort((-cd_comb, ranks_comb))
            
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]

        return X
