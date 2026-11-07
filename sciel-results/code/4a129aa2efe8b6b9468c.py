"""Multi-objective differential evolution optimizer for WFG7."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective DE with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin before 300s
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Smaller population for more generations
        N = 20
        F_de = 0.8  # differential evolution scaling factor
        CR = 0.5    # crossover rate

        # Initialize population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        def fast_non_dominated_sort(F):
            """
            Returns ranks (0-based) and fronts (list of lists of indices).
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Domination sets
            n_d = np.zeros(n, dtype=int)  # Domination counts
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # j dominates i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                
                if n_d[i] == 0:
                    rank[i] = 0 # Front 0

            current_front_idx = 0
            current_front = [i for i in range(n) if rank[i] == 0]
            
            fronts = []
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = current_front_idx + 1
                            next_front.append(j)
                current_front = next_front
                current_front_idx += 1
            
            return rank, fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for points with given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                order = np.argsort(F[indices, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin = F[indices[order[0]], m]
                fmax = F[indices[order[-1]], m]
                rng_m = fmax - fmin
                if rng_m < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (F[indices[order[k+1]], m] - F[indices[order[k-1]], m]) / rng_m
            return dist

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.perf_counter() > deadline:
                break

            # Differential evolution: for each individual, create trial
            X_new = np.empty_like(X)
            F_new = np.empty_like(F)
            
            for i in range(N):
                if used >= budget:
                    break
                if time.perf_counter() > deadline:
                    break

                # Select three distinct random parents different from i
                candidates = [j for j in range(N) if j != i]
                if len(candidates) < 3:
                    p1, p2, p3 = 0, 1, 2
                else:
                    p1, p2, p3 = rng.choice(candidates, size=3, replace=False)

                # Differential evolution mutation
                v = X[p1] + F_de * (X[p2] - X[p3])

                # Binomial crossover
                r = rng.random(dim)
                jrand = rng.integers(dim)
                trial = np.where(r < CR, v, X[i])
                trial[jrand] = v[jrand]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1
                X_new[i] = trial
                F_new[i] = f_trial
            
            # Combine parents and offspring
            X_comb = np.vstack([X, X_new])
            F_comb = np.vstack([F, F_new])
            
            # Fast Non-Dominated Sort
            rank, fronts = fast_non_dominated_sort(F_comb)
            
            # Select top N individuals
            X_sel = np.empty((N, dim))
            F_sel = np.empty((N, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    for idx in front:
                        X_sel[count] = X_comb[idx]
                        F_sel[count] = F_comb[idx]
                        count += 1
                else:
                    # Need to trim this front using crowding distance
                    need = N - count
                    cd = crowding_distance(F_comb, front)
                    # Sort front indices by CD descending
                    front_indices = np.array(front)
                    order = np.argsort(-cd)
                    for k in range(need):
                        idx = front_indices[order[k]]
                        X_sel[count] = X_comb[idx]
                        F_sel[count] = F_comb[idx]
                        count += 1
                    break
            
            X = X_sel
            F = F_sel

        return X
# EVOLVE-BLOCK-END
