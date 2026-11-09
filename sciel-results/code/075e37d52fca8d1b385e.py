"""Multi-objective optimizer for RE34 using efficient NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with tight budget management."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size: keep small to fit within budget
        pop_size = 20
        n_gen = max(1, (self.budget - pop_size) // pop_size)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def non_dominated_sort(F_arr):
            """Return list of fronts (lists of indices) and front indices array."""
            n = len(F_arr)
            if n == 0:
                return [], np.array([], dtype=int)

            # domination matrix: dom[i,j] = True if i dominates j
            dom = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dom[i, j] = True
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dom[j, i] = True

            # S[i] = indices dominated by i
            S = [np.where(dom[i])[0] for i in range(n)]
            # n_dom[i] = number of individuals that dominate i
            n_dom = dom.sum(axis=0)

            fronts = []
            front_idx = np.full(n, -1, dtype=int)
            current_front = np.where(n_dom == 0)[0]

            fi = 0
            while len(current_front) > 0:
                fronts.append(current_front.tolist())
                for i in current_front:
                    front_idx[i] = fi
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = np.array(next_front, dtype=int)
                fi += 1

            return fronts, front_idx

        def crowding_distance(F_arr, front_idx):
            """Compute crowding distance for each point."""
            n = len(F_arr)
            cd = np.zeros(n)
            max_front = front_idx.max() if n > 0 else 0

            for fi in range(max_front + 1):
                idx = np.where(front_idx == fi)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F_arr[idx, m]
                    order = np.argsort(vals)
                    cd[idx[order[0]]] = np.inf
                    cd[idx[order[-1]]] = np.inf
                    val_range = vals[order[-1]] - vals[order[0]]
                    if val_range > 1e-12:
                        for k in range(1, len(idx) - 1):
                            cd[idx[order[k]]] += (vals[order[k + 1]] - vals[order[k - 1]]) / val_range
            return cd

        def tournament_select(F_arr, front_idx, cd, k=2):
            """Select one parent via tournament selection."""
            n = len(F_arr)
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if front_idx[c] < front_idx[best]:
                    best = c
                elif front_idx[c] == front_idx[best]:
                    if cd[c] > cd[best]:
                        best = c
            return best

        def sbx_crossover(p1, p2):
            """Simulated Binary Crossover."""
            if rng.random() > 0.9:
                return p1.copy()
            child = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 15)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 15)
                    child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 15)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 15)
                    child[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
            return child

        def polynomial_mutation(x, eta=20):
            """Polynomial mutation."""
            child = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.9:
                    u = rng.random()
                    if u < 0.5:
                        mut = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        mut = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    child[d] += mut * span[d]
            return np.clip(child, lo, hi)

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Compute non-dominated sort and crowding for current population
            fronts, front_idx = non_dominated_sort(F)
            cd = crowding_distance(F, front_idx)

            # Generate offspring
            offspring_X = []
            offspring_F = []

            for _ in range(pop_size):
                if time.time() > deadline - 2:
                    break
                if used + len(offspring_X) >= self.budget:
                    break

                # Select parents
                p1_idx = tournament_select(F, front_idx, cd)
                p2_idx = tournament_select(F, front_idx, cd)

                # Crossover
                child = sbx_crossover(X[p1_idx], X[p2_idx])

                # Mutation
                child = polynomial_mutation(child)

                fc = problem(child)
                offspring_X.append(child)
                offspring_F.append(fc)
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts_all, front_idx_all = non_dominated_sort(F_all)
            cd_all = crowding_distance(F_all, front_idx_all)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for i in front:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                else:
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd_all[i], reverse=True)
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
