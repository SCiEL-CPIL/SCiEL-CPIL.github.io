"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        N = min(40, self.budget // 2)
        N = max(N, 10)

        # Initial population
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        def dominates(a, b):
            """True if a dominates b (all <= and at least one <)"""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # nd[i] = number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            fronts.pop()  # remove last empty list
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[indices, m]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                val_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if val_range > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / val_range
            return cd

        def tournament_select(X, F, k=2):
            """Tournament selection."""
            n = len(X)
            if n <= 1:
                return X[0].copy()
            candidates = rng.integers(0, n, size=k)
            # Pick the one with best front rank, then best crowding distance
            best = candidates[0]
            best_rank = fast_non_dominated_sort(F[candidates])[0]
            # Simplified: just pick randomly among candidates weighted by dominance
            # For efficiency, use simple tournament on non-dominated rank
            ranks = np.zeros(k, dtype=int)
            fronts = fast_non_dominated_sort(F[candidates])
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            best_idx = np.argmin(ranks)
            return X[candidates[best_idx]].copy()

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < N:
                if time.time() > deadline:
                    break
                # Select two parents via tournament
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)

                # SBX crossover
                child = p1.copy()
                if rng.random() < 0.9:  # crossover probability
                    eta_c = 15
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 1.0 / (eta_c + 1):
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            elif u >= 1.0 - 1.0 / (eta_c + 1):
                                beta = (2 * (1 - u)) ** (1.0 / (eta_c + 1))
                            else:
                                beta = 1.0
                            child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])

                # Polynomial mutation
                if rng.random() < 1.0 / self.dim:  # mutation probability
                    d = rng.integers(self.dim)
                    eta_m = 20
                    u = rng.random()
                    if u <= 1.0 / (eta_m + 1):
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    elif u >= 1.0 - 1.0 / (eta_m + 1):
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    else:
                        delta_q = 0.0
                    child[d] = child[d] + delta_q * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_comb)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(next_X)
                    cd = crowding_distance(F_comb, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
