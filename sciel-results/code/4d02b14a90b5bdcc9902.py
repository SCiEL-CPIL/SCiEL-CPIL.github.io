"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Population size: use a reasonable fraction of budget
        self.pop_size = min(100, max(20, budget // 2))
        self.max_evals = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Initial population
        pop = self.pop_size
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop

        # Helper: non-dominated sort
        def non_dominated_sort(F_arr):
            """Return list of fronts (list of indices)."""
            n = len(F_arr)
            if n == 0:
                return []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distances for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            dist = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                vals = F_arr[front_indices, k]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range < 1e-12:
                    continue
                for i in range(1, n - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return dist

        def fast_non_dominated_sort_and_crowd(F_arr):
            """Return rank array and crowding distance array."""
            fronts = non_dominated_sort(F_arr)
            n = len(F_arr)
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            crowd = np.zeros(n)
            for front in fronts:
                cd = crowding_distance(F_arr, front)
                for i, idx in enumerate(front):
                    crowd[idx] = cd[i]
            return rank, crowd

        def tournament_select(rank, crowd, k=2):
            """Tournament selection based on rank then crowding."""
            candidates = rng.integers(len(rank), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and crowd[c] > crowd[best]:
                    best = c
            return best

        # Evolution loop
        while used < self.max_evals:
            # Check time
            if time.time() > deadline:
                break

            rank, crowd = fast_non_dominated_sort_and_crowd(F)

            # Generate offspring
            offspring_size = pop
            offspring = np.empty((offspring_size, self.dim))
            for i in range(offspring_size):
                p1 = tournament_select(rank, crowd)
                p2 = tournament_select(rank, crowd)
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20
                    child1 = np.empty(self.dim)
                    child2 = np.empty(self.dim)
                    for d in range(self.dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x1 = X[p1, d]
                        x2 = X[p2, d]
                        child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        child2[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    offspring[i] = child1
                else:
                    offspring[i] = X[p1].copy()

                # Polynomial mutation
                if rng.random() < 1.0 / self.dim:
                    d = rng.integers(self.dim)
                    eta_m = 20
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    span = hi[d] - lo[d]
                    offspring[i, d] += delta * span
                    offspring[i, d] = np.clip(offspring[i, d], lo[d], hi[d])

            # Evaluate offspring
            for i in range(offspring_size):
                if time.time() > deadline:
                    break
                offspring[i] = np.clip(offspring[i], lo, hi)
                F[used + i] if used + i < len(F) else None
                # We need to handle this more carefully
                break  # This loop structure is wrong, let me fix

            # Let me restructure the offspring evaluation
            # Actually let me redo this more carefully

            # Evaluate offspring
            F_offspring = np.empty((offspring_size, self.n_obj))
            for i in range(offspring_size):
                if time.time() > deadline:
                    break
                offspring[i] = np.clip(offspring[i], lo, hi)
                F_offspring[i] = problem(offspring[i])
                used += 1

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring[:used - pop]])
            F_combined = np.vstack([F, F_offspring[:used - pop]])

            # Selection
            rank_c, crowd_c = fast_non_dominated_sort_and_crowd(F_combined)

            # Sort by rank then crowding
            order = np.lexsort((np.argsort(-crowd_c), rank_c))
            X = X_combined[order[:pop]]
            F = F_combined[order[:pop]]

            if used >= self.max_evals:
                break

        return X
# EVOLVE-BLOCK-END
