"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a moderate fraction of budget
        pop_size = max(20, min(50, self.budget // 4))
        # Ensure we don't exceed budget
        if pop_size * 2 > self.budget:
            pop_size = max(10, self.budget // 4)

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    rank[i] = 0
                    if not fronts or len(fronts) == 0:
                        fronts.append([])
                    fronts[0].append(i)

            i = 0
            while i < len(fronts):
                current_front = fronts[i]
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dominated[q] -= 1
                        if n_dominated[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            distances = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[i][m] for i in indices]
                min_val = min(vals)
                max_val = max(vals)
                if max_val - min_val < 1e-15:
                    distances[indices.index(indices[0])] = float('inf')
                    continue
                sorted_idx = sorted(range(n), key=lambda k: vals[k])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        def tournament_select(fronts, F, X):
            """Tournament selection based on rank and crowding distance."""
            # Compute ranks
            ranks = [0] * len(F)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances per front
            cd = [0.0] * len(F)
            for front in fronts:
                cds = crowding_distance(F, front)
                for idx, d in zip(front, cds):
                    cd[idx] = d

            # Tournament
            i1, i2 = rng.integers(len(F), size=2)
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                if cd[i1] > cd[i2]:
                    return i1
                else:
                    return i2

        def crossover(x1, x2):
            """SBX crossover."""
            eta = 20.0
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            return c1, c2

        def mutation(x, sigma=0.1):
            """Polynomial mutation."""
            eta = 20.0
            child = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[i] += sigma * delta_q
            return np.clip(child, lo, hi)

        # Main loop
        while used < self.budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < pop_size and used + len(offspring_X) < self.budget:
                p1 = tournament_select(fronts, F, X)
                p2 = tournament_select(fronts, F, X)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used < self.budget:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to sort by crowding distance
                    cds = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cds[front.index(i)], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
