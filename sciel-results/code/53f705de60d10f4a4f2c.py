"""Rewritten program using NSGA-II style multi-objective evolutionary optimizer."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II inspired multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Population size: keep it manageable but diverse
        pop_size = min(40, max(10, budget // 5))
        if pop_size * 2 > budget:
            pop_size = max(5, budget // 4)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(fitnesses):
            """Return list of fronts (each front is a list of indices)."""
            m = len(fitnesses)
            S = [[] for _ in range(m)]
            n_dom = [0] * m
            rank = [0] * m
            fronts = []
            front0 = []
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    front0.append(i)
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(fitnesses, indices):
            """Compute crowding distance for given indices."""
            m = len(indices)
            if m <= 2:
                return [float('inf')] * m
            cd = [0.0] * m
            for d in range(fitnesses.shape[1]):
                obj = np.array([fitnesses[i, d] for i in indices])
                order = np.argsort(obj)
                cd_sorted = [0.0] * m
                cd_sorted[order[0]] = float('inf')
                cd_sorted[order[-1]] = float('inf')
                obj_range = obj[order[-1]] - obj[order[0]]
                if obj_range > 1e-12:
                    for i in range(1, m - 1):
                        cd_sorted[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / obj_range
                else:
                    for i in range(1, m - 1):
                        cd_sorted[order[i]] += float('inf')
                for i in range(m):
                    cd[i] += cd_sorted[i]
            return cd

        def tournament_select(pop_indices, fitnesses, fronts, k=2):
            """Binary tournament selection."""
            a, b = rng.choice(pop_indices, size=2, replace=False)
            ra, rb = fronts[a], fronts[b]
            if ra < rb:
                return a
            elif ra > rb:
                return b
            # Same rank: use crowding distance
            cd = crowding_distance(fitnesses, [a, b])
            return a if cd[0] >= cd[1] else b

        def crossover(x1, x2):
            """SBX-like crossover with uniform mixing."""
            if rng.random() < 0.5:
                alpha = rng.random()
                c1 = alpha * x1 + (1 - alpha) * x2
                c2 = (1 - alpha) * x1 + alpha * x2
            else:
                c1 = x1.copy()
                c2 = x2.copy()
            return c1, c2

        def mutate(x, sigma=0.1):
            """Gaussian mutation."""
            x = x + sigma * (hi - lo) * rng.standard_normal(n)
            return np.clip(x, lo, hi)

        # Main evolutionary loop
        while used < budget:
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)
            # Build rank array
            rank_arr = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank_arr[idx] = r

            # Create offspring
            n_off = min(pop_size, budget - used)
            offspring = []
            for _ in range(n_off):
                p1 = tournament_select(np.arange(pop_size), F, rank_arr)
                p2 = tournament_select(np.arange(pop_size), F, rank_arr)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < n_off:
                    offspring.append(c2)
                if used + len(offspring) >= budget:
                    break

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring], dtype=float)
            used += len(offspring)

            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring)])
            F_all = np.vstack([F, F_off])
            total = len(X_all)

            # Non-dominated sort on combined
            fronts_all = fast_non_dominated_sort(F_all)
            rank_all = np.zeros(total, dtype=int)
            for r, front in enumerate(fronts_all):
                for idx in front:
                    rank_all[idx] = r

            # Select next population
            next_X = []
            next_F = []
            for front in fronts_all:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_all, front)
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
