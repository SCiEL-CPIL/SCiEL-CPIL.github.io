"""Multi-objective optimizer for WFG9 with NSGA-II style search and deadline guard."""
import numpy as np
import time


class Optimizer:
    """NSGA-II inspired multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(100, max(20, budget // 2))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        # NSGA-II helper functions
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = [0] * n  # number of individuals dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1

            for i in range(n):
                if n_count[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts[:-1]  # drop last empty front

        def crowding_distance(F, front_indices):
            """Compute crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def sort_by_dominance_and_crowding(F):
            """Return indices sorted by non-dominated rank then crowding distance."""
            fronts = fast_non_dominated_sort(F)
            order = []
            for front in fronts:
                if not front:
                    continue
                cd = crowding_distance(F, front)
                # Sort by crowding distance descending within each front
                front_sorted = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                order.extend(front_sorted)
            return order

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort current population
            order = sort_by_dominance_and_crowding(F)

            # Select parents using tournament selection
            parents = []
            for _ in range(pop_size):
                # Binary tournament
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                # Find rank
                rank_map = {}
                for rank, front in enumerate(fast_non_dominated_sort(F)):
                    for idx in front:
                        rank_map[idx] = rank
                r1 = rank_map.get(i1, 999)
                r2 = rank_map.get(i2, 999)
                if r1 < r2:
                    parents.append(i1)
                elif r2 < r1:
                    parents.append(i2)
                else:
                    # Same rank, use crowding
                    cd = crowding_distance(F, [i1, i2])
                    parents.append(i1 if cd[0] >= cd[1] else i2)

            # Create offspring
            children_X = np.zeros((pop_size, dim))
            children_F = np.zeros((pop_size, n_obj))

            # Adaptive mutation step: start large, shrink over time
            progress = used / max(1, budget)
            base_step = 0.3 * (1 - progress * 0.7)  # shrinks from 0.3 to 0.09

            for i in range(pop_size):
                if used + i >= budget:
                    break

                p1 = parents[i]
                p2 = parents[(i + 1) % pop_size]

                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    x1, x2 = X[p1].copy(), X[p2].copy()
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            # SBX
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                            child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                            child[d] = 0.5 * ((1 + beta) * x2[d] + (1 - beta) * x1[d])
                else:
                    child = X[p1].copy()

                # Mutation
                if rng.random() < 0.9:
                    step = base_step * rng.standard_normal(dim)
                    child = child + step * (hi - lo) / 2

                child = np.clip(child, lo, hi)
                children_X[i] = child

            # Evaluate children (respect budget)
            max_children = min(pop_size, budget - used)
            for i in range(max_children):
                children_F[i] = problem(children_X[i])
                used += 1
                if time.time() > deadline:
                    break

            # Combine parent and child populations
            X_comb = np.vstack([X, children_X[:max_children]])
            F_comb = np.vstack([F, children_F[:max_children]])

            # Select next generation
            order_comb = sort_by_dominance_and_crowding(F_comb)
            X = X_comb[order_comb[:pop_size]]
            F = F_comb[order_comb[:pop_size]]

            # Record for hypervolume
            all_X.append(X.copy())
            all_F.append(F.copy())

        # Return the best non-dominated set found
        return X
