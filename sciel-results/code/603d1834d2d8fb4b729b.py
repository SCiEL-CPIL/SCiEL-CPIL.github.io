"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 4))
        if pop_size % 2 == 1:
            pop_size -= 1

        # Random initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [set() for _ in range(n)]
            nd = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        nd[i] += 1

            current_front = [i for i in range(n) if nd[i] == 0]
            if not current_front:
                return [list(range(n))]
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)

            n = len(front_indices)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = np.array([F[i][m] for i in front_indices])
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 1e-15:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / val_range
            return dist

        def sort_by_dominance(F):
            """Sort indices by non-dominated rank, then crowding distance."""
            fronts = fast_non_dominated_sort(F)
            order = []
            for front in fronts:
                cd = crowding_distance(F, front)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                order.extend(sorted_front)
            return order

        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)

            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill remaining with copies of best
                    offspring_X[i:] = X[:pop_size - i]
                    offspring_F[i:] = F[:pop_size - i]
                    break

                # Tournament selection (size 2)
                a = np.random.randint(0, pop_size)
                b = np.random.randint(0, pop_size)
                # Select non-dominated one
                if np.all(F[a] <= F[b]) and np.any(F[a] < F[b]):
                    p1 = X[a]
                elif np.all(F[b] <= F[a]) and np.any(F[b] < F[a]):
                    p1 = X[b]
                else:
                    p1 = X[a if np.random.rand() < 0.5 else b]

                # SBX crossover
                if np.random.rand() < 0.9:
                    # Second parent
                    c = np.random.randint(0, pop_size)
                    d = np.random.randint(0, pop_size)
                    if np.all(F[c] <= F[d]) and np.any(F[c] < F[d]):
                        p2 = X[c]
                    elif np.all(F[d] <= F[c]) and np.any(F[d] < F[c]):
                        p2 = X[d]
                    else:
                        p2 = X[c if np.random.rand() < 0.5 else d]

                    eta = 20.0
                    child1 = np.empty(dim)
                    child2 = np.empty(dim)
                    for j in range(dim):
                        if np.random.rand() < 0.5:
                            if abs(p1[j] - p2[j]) < 1e-14:
                                child1[j] = p1[j]
                                child2[j] = p2[j]
                            else:
                                r = np.random.rand()
                                if r <= 0.5:
                                    beta = (2 * r) ** (1.0 / (eta + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta + 1))
                                child1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                                child2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                        else:
                            child1[j] = p1[j]
                            child2[j] = p2[j]

                    # Polynomial mutation
                    for j in range(dim):
                        if np.random.rand() < 1.0 / dim:
                            r = np.random.rand()
                            if r < 0.5:
                                delta = (2 * r) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - r)) ** (1.0 / (eta + 1))
                            child1[j] += delta * (hi[j] - lo[j])
                        if np.random.rand() < 1.0 / dim:
                            r = np.random.rand()
                            if r < 0.5:
                                delta = (2 * r) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - r)) ** (1.0 / (eta + 1))
                            child2[j] += delta * (hi[j] - lo[j])

                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)

                    offspring_X[i] = child1
                    offspring_F[i] = problem(child1)
                    used += 1

                    if i + 1 < pop_size and used < budget:
                        offspring_X[i + 1] = child2
                        offspring_F[i + 1] = problem(child2)
                        used += 1
                        i += 1
                else:
                    # Just mutation
                    child = p1.copy()
                    for j in range(dim):
                        if np.random.rand() < 1.0 / dim:
                            r = np.random.rand()
                            if r < 0.5:
                                delta = (2 * r) ** (1.0 / (21.0)) - 1
                            else:
                                delta = 1 - (2 * (1 - r)) ** (1.0 / (21.0))
                            child[j] += delta * (hi[j] - lo[j])
                    child = np.clip(child, lo, hi)
                    offspring_X[i] = child
                    offspring_F[i] = problem(child)
                    used += 1

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Sort combined population
            order = sort_by_dominance(combined_F)

            # Select top pop_size
            X = combined_X[order[:pop_size]]
            F = combined_F[order[:pop_size]]

        return X
# EVOLVE-BLOCK-END
