"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
                if nd[i] == 0:
                    current_front.append(i)

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Any remaining points (shouldn't happen)
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F, indices):
            """Returns crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                sorted_idx = np.argsort(vals)
                cd_sorted = np.zeros(n)
                if vals[sorted_idx[-1]] > vals[sorted_idx[0]]:
                    cd_sorted[0] = np.inf
                    cd_sorted[-1] = np.inf
                    for k in range(1, n - 1):
                        cd_sorted[k] = (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (
                            vals[sorted_idx[-1]] - vals[sorted_idx[0]] + 1e-30)
                # Map back
                for i, si in enumerate(sorted_idx):
                    cd[si] += cd_sorted[i]
            return cd

        def tournament_select(F, X, k=2):
            """Tournament selection based on non-domination rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = crowding_distance(F, front)

            # Tournament
            candidates = rng.integers(0, len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                    best = c
            return X[best].copy()

        def crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                eta = 20
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20
            x_new = x.copy()
            for i in range(dim):
                if rng.random() < 0.9:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(x_new, lo, hi)

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring // 2):
                if time.time() > deadline:
                    break
                p1 = tournament_select(F, X)
                p2 = tournament_select(F, X)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.extend([c1, c2])
                offspring_F.extend([problem(c1), problem(c2)])
                used += 2

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(all_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need to sort by crowding distance
                    cd = crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:pop_size - len(new_X)]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END
