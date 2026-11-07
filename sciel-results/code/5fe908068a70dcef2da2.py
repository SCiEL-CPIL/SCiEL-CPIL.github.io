"""Multi-objective optimizer on WFG9 using NSGA-II with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective evolutionary optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance diversity and computation
        pop_size = min(100, max(30, budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = np.zeros(n, dtype=int)
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1

            for i in range(n):
                if n_doms[i] == 0:
                    current_front.append(i)

            fronts.append(current_front)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distances for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            f = F[front_indices]

            for m in range(n_obj):
                sorted_idx = np.argsort(f[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = f[sorted_idx[-1], m] - f[sorted_idx[0], m]
                if f_range > 1e-15:
                    for i in range(1, n - 1):
                        distances[sorted_idx[i]] += (f[sorted_idx[i + 1], m] - f[sorted_idx[i - 1], m]) / f_range

            return distances

        def select_parents(fronts, F, pop_size):
            """Select parents using tournament selection on fronts and crowding."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Compute crowding distance per front
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        crowd[idx] = cd[j]

            # Tournament selection
            parents = []
            while len(parents) < pop_size:
                # Pick two random individuals
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if rank[a] < rank[b]:
                    winner = a
                elif rank[b] < rank[a]:
                    winner = b
                else:
                    winner = a if crowd[a] > crowd[b] else b
                parents.append(winner)
            return parents

        def crossover(x1, x2):
            """SBX crossover."""
            eta = 20.0
            child1 = np.zeros(n_dim)
            child2 = np.zeros(n_dim)
            for i in range(n_dim):
                if rng.random() < 0.5:
                    beta1 = 1.0 / (2.0 * (rng.random() + 1e-15))
                    beta1 = beta1 ** (1.0 / (eta + 1.0))
                    beta2 = 1.0 / (2.0 * (rng.random() + 1e-15))
                    beta2 = beta2 ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta1) * x1[i] + (1 - beta1) * x2[i])
                    child2[i] = 0.5 * ((1 + beta2) * x2[i] + (1 - beta2) * x1[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(x, rate=0.1, sigma=0.1):
            """Polynomial mutation."""
            eta = 20.0
            child = x.copy()
            for i in range(n_dim):
                if rng.random() < rate:
                    delta1 = (rng.random() * 2) ** (1.0 / (eta + 1.0)) - 1.0
                    delta2 = 1.0 - (rng.random() * 2) ** (1.0 / (eta + 1.0))
                    child[i] += sigma * (hi[i] - lo[i]) * (delta1 + delta2)
            return np.clip(child, lo, hi)

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Select parents
            parents = select_parents(fronts, F, pop_size)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size - used if used + pop_size > budget else pop_size
            n_offspring = min(n_offspring, budget - used)

            for _ in range(n_offspring // 2):
                p1 = parents[rng.integers(len(parents))]
                p2 = parents[rng.integers(len(parents))]
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                used += 2
                if used >= budget or time.time() > deadline:
                    break

            # Evaluate offspring
            for ox in offspring_X:
                offspring_F.append(problem(ox))
            if offspring_F:
                offspring_F = np.array(offspring_F)
                offspring_X = np.array(offspring_X)

                # Combine parent and offspring
                combined_X = np.vstack([X, offspring_X])
                combined_F = np.vstack([F, offspring_F])

                # Select next generation
                combined_fronts = fast_non_dominated_sort(combined_F)

                next_X = []
                next_F = []
                for front in combined_fronts:
                    if len(next_X) + len(front) <= pop_size:
                        for idx in front:
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    else:
                        # Need to truncate this front
                        remaining = pop_size - len(next_X)
                        cd = crowding_distance(combined_F, front)
                        sorted_front = sorted(range(len(front)), key=lambda i: -cd[i])
                        for j in range(remaining):
                            idx = front[sorted_front[j]]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                        break

                X = np.array(next_X)
                F = np.array(next_F)
            else:
                # No offspring generated, break
                break

        return X
# EVOLVE-BLOCK-END
