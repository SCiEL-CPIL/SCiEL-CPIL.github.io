"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer using NSGA-II-like selection with
    fast non-dominated sorting and crowding distance, plus a time-bounded
    safety margin."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit with 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(10, self.budget // 5))
        n_eval = 0

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        n_eval += pop_size

        # Helper: fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1  # wait, this is wrong direction
                # Actually let me redo this properly

            # Redo: S[i] = set of solutions dominated by i
            # nd[i] = number of solutions that dominate i
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Does i dominate j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Does j dominate i?
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
                    rank_i = 0

            # Build rank array
            rank = np.full(n, -1, dtype=int)
            for i in fronts[0]:
                rank[i] = 0

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return rank, fronts[:-1]

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = len(indices)
            cd = np.zeros(m)
            n_obj = F.shape[1]
            for k in range(n_obj):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if f_range > 1e-15:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (
                            obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]
                        ) / f_range
            return cd

        # Helper: binary tournament selection
        def tournament_select(rank, cd, X, F, n_select):
            selected_X = []
            selected_F = []
            for _ in range(n_select):
                i, j = rng.integers(len(rank), size=2)
                # Compare by rank first, then crowding distance
                if rank[i] < rank[j]:
                    winner = i
                elif rank[j] < rank[i]:
                    winner = j
                else:
                    # Same rank, use crowding distance
                    # Need to compute cd for all
                    cd_all = crowding_distance(F, np.arange(len(F)))
                    if cd_all[i] > cd_all[j]:
                        winner = i
                    else:
                        winner = j
                selected_X.append(X[winner])
                selected_F.append(F[winner])
            return np.array(selected_X), np.array(selected_F)

        # Helper: crossover and mutation
        def crossover(x1, x2):
            # Simulated binary crossover
            if rng.random() < 0.9:  # crossover probability
                eta = 20.0
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for i in range(self.dim):
                    if x1[i] == x2[i]:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            # Polynomial mutation
            eta = 20.0
            x_new = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.8:  # mutation probability
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_new[i] += delta * (hi[i] - lo[i])
            return np.clip(x_new, lo, hi)

        # Main loop
        while n_eval < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Evaluate current population
            rank, fronts = fast_non_dominated_sort(F)

            # Create mating pool via tournament selection
            n_parents = min(pop_size, self.budget - n_eval)
            if n_parents <= 0:
                break

            # Select parents
            parents_X = []
            for _ in range(n_parents):
                if time.time() > deadline:
                    break
                i, j = rng.integers(len(rank), size=2)
                if rank[i] < rank[j]:
                    winner = i
                elif rank[j] < rank[i]:
                    winner = j
                else:
                    # Same rank, use crowding distance
                    cd_all = crowding_distance(F, np.arange(len(F)))
                    if cd_all[i] > cd_all[j]:
                        winner = i
                    else:
                        winner = j
                parents_X.append(X[winner])

            if not parents_X:
                break

            parents_X = np.array(parents_X)

            # Generate offspring
            offspring_X = []
            for _ in range(n_parents):
                if time.time() > deadline:
                    break
                if n_eval >= self.budget:
                    break
                # Pick two parents
                p1_idx = rng.integers(len(parents_X))
                p2_idx = rng.integers(len(parents_X))
                c1, c2 = crossover(parents_X[p1_idx], parents_X[p2_idx])
                c1 = mutate(c1)
                offspring_X.append(c1)
                n_eval += 1
                # Optionally add second child
                if n_eval < self.budget and rng.random() < 0.5:
                    c2 = mutate(c2)
                    offspring_X.append(c2)
                    n_eval += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])

            # Combine parent and offspring populations
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Select next generation
            rank_comb, fronts_comb = fast_non_dominated_sort(F_comb)

            # Sort fronts by rank
            next_X = []
            next_F = []
            for front in fronts_comb:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        cd_front = crowding_distance(F_comb[front], front)
                        sorted_front = np.argsort(-cd_front)
                        for k in range(remaining):
                            idx = front[sorted_front[k]]
                            next_X.append(X_comb[idx])
                            next_F.append(F_comb[idx])
                    break

            if not next_X:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
