"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # NSGA-II parameters
        pop_size = 50  # larger population for better coverage
        crossover_prob = 0.9
        mutation_prob = 0.1
        eta_c = 20.0  # distribution index for crossover
        eta_m = 20.0  # distribution index for mutation

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F, N):
            """NSGA-II non-dominated sorting. Returns list of fronts (lists of indices)."""
            S = [[] for _ in range(N)]
            n = np.zeros(N, dtype=int)
            rank = np.zeros(N, dtype=int)
            front = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[i].append(j)
                    elif np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        n[i] += 1
                if n[i] == 0:
                    rank[i] = 0
                    front.append(i)

            fronts = [front]
            k = 0
            while front:
                k += 1
                next_front = []
                for i in front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            rank[j] = k
                            next_front.append(j)
                fronts.append(next_front)
                front = next_front

            if fronts and len(fronts[-1]) == 0:
                fronts.pop()
            return fronts, rank

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            cd = [0.0] * n
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                fmin = F[front_indices[sorted_idx[0]]][m]
                fmax = F[front_indices[sorted_idx[-1]]][m]
                if fmax - fmin < 1e-10:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F[front_indices[sorted_idx[i + 1]]][m] - F[front_indices[sorted_idx[i - 1]]][m]) / (fmax - fmin)
            return cd

        def tournament_select(fronts, rank, F, pop_size, n_tour=2):
            """Tournament selection based on rank and crowding distance."""
            # Compute crowding distance for all points
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    front_cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = front_cd[i]

            def compare(i, j):
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                else:
                    return i if cd[i] > cd[j] else j

            selected = []
            for _ in range(pop_size):
                candidates = rng.integers(0, pop_size, size=n_tour)
                best = candidates[0]
                for c in candidates[1:]:
                    best = compare(best, c)
                selected.append(best)
            return selected

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() > crossover_prob:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def polynomial_mutation(x):
            """Polynomial mutation."""
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < mutation_prob:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    range_i = hi[i] - lo[i]
                    x_mut[i] += delta_q * range_i
            return np.clip(x_mut, lo, hi)

        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate current population
            fronts, rank = fast_non_dominated_sort(F, pop_size)

            # Selection
            parents = tournament_select(fronts, rank, F, pop_size)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            for i in range(0, pop_size, 2):
                p1, p2 = parents[i], parents[i + 1]
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if used >= budget:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            N_combined = len(combined_F)

            # Non-dominated sort on combined
            fronts_c, rank_c = fast_non_dominated_sort(combined_F, N_combined)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts_c:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    front_cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: front_cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
