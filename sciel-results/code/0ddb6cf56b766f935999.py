"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        rng = np.random.default_rng(0)

        # Population size - balance between diversity and convergence
        pop_size = 50

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting (NSGA-II)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of solutions dominating i
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_count[i] += 1

            front = [i for i in range(n) if n_count[i] == 0]
            current_rank = 0
            while front:
                next_front = []
                for i in front:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                current_rank += 1
                front = next_front
            return rank

        def crowding_distance(F, rank):
            """Calculate crowding distance for each solution."""
            n = len(F)
            dist = np.zeros(n)
            unique_ranks = np.unique(rank)

            for r in unique_ranks:
                members = np.where(rank == r)[0]
                if len(members) <= 2:
                    dist[members] = np.inf
                    continue

                for m in range(n_obj):
                    obj_vals = F[members, m]
                    min_val = obj_vals.min()
                    max_val = obj_vals.max()
                    if max_val == min_val:
                        continue
                    sorted_idx = np.argsort(obj_vals)
                    dist[members[sorted_idx[0]]] = np.inf
                    dist[members[sorted_idx[-1]]] = np.inf
                    for k in range(1, len(members) - 1):
                        dist[members[sorted_idx[k]]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)

            return dist

        def tournament_select(X, F, rank, dist, k=2):
            """Tournament selection based on rank and crowding distance."""
            indices = rng.integers(len(X), size=k)
            best = indices[0]
            for i in indices[1:]:
                if rank[i] < rank[best] or (rank[i] == rank[best] and dist[i] > dist[best]):
                    best = i
            return X[best].copy()

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sorting and crowding distance
            rank = fast_non_dominated_sort(F)
            dist = crowding_distance(F, rank)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring):
                # Differential evolution style mutation
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                p3 = tournament_select(X, F, rank, dist)

                # DE/3 mutation
                F_mut = 0.8
                child = p1 + F_mut * (p2 - p3)
                child = np.clip(child, lo, hi)

                # Crossover (binomial)
                crossover_rate = 0.9
                mask = rng.random(dim) < crossover_rate
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                child = np.where(mask, child, p1)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= budget or time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            combined_rank = fast_non_dominated_sort(combined_F)
            combined_dist = crowding_distance(combined_F, combined_rank)

            # Sort by rank, then by crowding distance
            sort_order = np.lexsort((combined_dist, combined_rank))
            next_X = combined_X[sort_order[:pop_size]]
            next_F = combined_F[sort_order[:pop_size]]

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END
