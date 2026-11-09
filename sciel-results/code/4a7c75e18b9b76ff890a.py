import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9 (3 objectives, 12 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Budget allocation
        pop_size = 40
        n_gen = max(1, (self.budget - pop_size) // pop_size)
        # Ensure we don't exceed budget
        total_evals = pop_size + n_gen * pop_size
        if total_evals > self.budget:
            n_gen = (self.budget - pop_size) // pop_size
            total_evals = pop_size + n_gen * pop_size

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            u = rng.random(self.dim)
            child1 = np.empty(self.dim)
            child2 = np.empty(self.dim)
            for i in range(self.dim):
                if u[i] <= 0.5:
                    beta = (2 * u[i]) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u[i]))) ** (1.0 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial mutation
        def polynomial_mutate(x, eta=20.0):
            u = rng.random(self.dim)
            mutated = x.copy()
            for i in range(self.dim):
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1.0 / (eta + 1))
                mutated[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(mutated, lo, hi)

        # Tournament selection (binary, by rank then crowding)
        def tournament_select(ranks, crowd, k=2):
            idx1 = rng.integers(len(ranks))
            idx2 = rng.integers(len(ranks))
            if ranks[idx1] < ranks[idx2]:
                return idx1
            elif ranks[idx2] < ranks[idx1]:
                return idx2
            else:
                return idx1 if crowd[idx1] > crowd[idx2] else idx2

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = [set()]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    ranks[i] = 0
                    front[0].add(i)

            k = 0
            while len(front[k]) > 0:
                next_front = set()
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            ranks[j] = k + 1
                            next_front.add(j)
                k += 1
                if next_front:
                    front.append(next_front)
                else:
                    break

            return ranks

        # Crowding distance
        def crowding_distance(F, rank_mask):
            n = len(F)
            crowd = np.zeros(n)
            indices = np.where(rank_mask)[0]
            m = len(indices)
            if m <= 2:
                crowd[indices] = np.inf
                return crowd
            for obj in range(self.n_obj):
                sorted_idx = indices[np.argsort(F[indices, obj])]
                crowd[sorted_idx[0]] = np.inf
                crowd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], obj] - F[sorted_idx[0], obj]
                if f_range < 1e-15:
                    continue
                for i in range(1, m - 1):
                    crowd[sorted_idx[i]] += (F[sorted_idx[i + 1], obj] - F[sorted_idx[i - 1], obj]) / f_range
            return crowd

        # Main NSGA-II loop
        for gen in range(n_gen):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Evaluate current population ranks and crowding
            ranks = non_dominated_sort(F)
            crowd = np.zeros(pop_size)
            for r in range(max(ranks) + 1):
                mask = (ranks == r)
                if np.sum(mask) >= pop_size:
                    # Last front, no truncation needed
                    crowd[mask] = np.inf
                else:
                    crowd[mask] = crowding_distance(F, mask)

            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                p1 = tournament_select(ranks, crowd)
                p2 = tournament_select(ranks, crowd)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X.append(c1)
                used += 1
                if used < self.budget and len(offspring_X) < pop_size:
                    offspring_X.append(c2)
                    used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            n_off = len(offspring_X)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)

            combined_ranks = non_dominated_sort(combined_F)
            combined_crowd = np.zeros(n_combined)
            for r in range(max(combined_ranks) + 1):
                mask = (combined_ranks == r)
                if np.sum(mask) >= pop_size:
                    combined_crowd[mask] = np.inf
                else:
                    combined_crowd[mask] = crowding_distance(combined_F, mask)

            # Sort by rank then crowding
            sort_order = np.lexsort((combined_crowd, combined_ranks))
            selected = sort_order[:pop_size]

            X = combined_X[selected]
            F = combined_F[selected]

        return X
