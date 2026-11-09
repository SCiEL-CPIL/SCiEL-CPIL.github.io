"""Multi-objective optimizer for RE34 using a fast NSGA-II variant."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an efficient NSGA-II style approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size: balance between diversity and convergence
        pop_size = min(60, max(20, self.budget // 3))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            if n == 0:
                return []

            # For each point, store the set of points it dominates
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)  # number of points that dominate i

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            fronts = []

            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = self.n_obj

            for k in range(m):
                # Sort by objective k
                vals = F[front_indices, k]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf

                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 1e-12:
                    for i in range(1, n - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range

            return cd

        def tournament_select(X, F, front_idx, cd, k=2):
            """Tournament selection based on front rank and crowding distance."""
            candidates = rng.integers(len(X), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if front_idx[c] < front_idx[best]:
                    best = c
                elif front_idx[c] == front_idx[best] and cd[c] > cd[best]:
                    best = c
            return X[best].copy()

        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            """Simulated Binary Crossover."""
            child = np.zeros_like(x1)
            for d in range(len(x1)):
                if rng.random() < 0.5:
                    # Compute beta
                    if x1[d] == x2[d]:
                        beta = 1.0
                    else:
                        if x1[d] < x2[d]:
                            x_small, x_big = x1[d], x2[d]
                        else:
                            x_small, x_big = x2[d], x1[d]

                        rand = rng.random()
                        if rand <= 0.5:
                            beta = (2 * rand) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - rand))) ** (1 / (eta + 1))

                        if rng.random() < 0.5:
                            child[d] = 0.5 * ((1 + beta) * x_small + (1 - beta) * x_big)
                        else:
                            child[d] = 0.5 * ((1 - beta) * x_small + (1 + beta) * x_big)
                else:
                    child[d] = x1[d]
            return np.clip(child, lo, hi)

        def polynomial_mutation(x, lo, hi, eta=20.0):
            """Polynomial mutation."""
            x_mut = x.copy()
            for d in range(len(x)):
                if rng.random() < 1.0 / len(x):
                    delta = rng.random()
                    if delta < 0.5:
                        mut = (2 * delta) ** (1 / (eta + 1)) - 1
                    else:
                        mut = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
                    x_mut[d] += mut * (hi[d] - lo[d])
            return np.clip(x_mut, lo, hi)

        # Main optimization loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            # Compute current front indices and crowding distances
            fronts = fast_non_dominated_sort(F)
            front_idx = np.full(pop_size, -1, dtype=int)
            for fi, front in enumerate(fronts):
                for i in front:
                    front_idx[i] = fi

            # Compute crowding distance for all points
            cd_all = np.zeros(pop_size)
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for j, i in enumerate(front):
                    cd_all[i] = cd_front[j]

            for _ in range(pop_size):
                if time.time() > deadline - 2:
                    break

                # Select parents
                p1 = tournament_select(X, F, front_idx, cd_all)
                p2 = tournament_select(X, F, front_idx, cd_all)

                # Crossover
                if rng.random() < 0.9:
                    child = sbx_crossover(p1, p2, lo, hi)
                else:
                    child = p1.copy()

                # Mutation
                child = polynomial_mutation(child, lo, hi)

                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            n_all = len(X_all)

            # Fast non-dominated sort on combined population
            fronts_all = fast_non_dominated_sort(F_all)

            # Select next generation
            new_X = []
            new_F = []
            front_idx_all = np.full(n_all, -1, dtype=int)
            for fi, front in enumerate(fronts_all):
                for i in front:
                    front_idx_all[i] = fi

            cd_all_new = np.zeros(n_all)
            for front in fronts_all:
                cd_front = crowding_distance(F_all, front)
                for j, i in enumerate(front):
                    cd_all_new[i] = cd_front[j]

            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for i in front:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                else:
                    # Sort by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda i: cd_all_new[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for i in front_sorted[:needed]:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
