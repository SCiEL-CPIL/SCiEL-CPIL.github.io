"""Multi-objective optimizer for RE21 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(40, max(10, self.budget // 5))

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Helper: compute crowding distance
        def crowding_distance(f):
            n = len(f)
            if n <= 2:
                return np.ones(n)
            cd = np.zeros(n)
            for m in range(n_obj):
                order = np.argsort(f[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                fmin, fmax = f[order[0], m], f[order[-1], m]
                if fmax - fmin < 1e-15:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (f[order[i + 1], m] - f[order[i - 1], m]) / (fmax - fmin)
            return cd

        # Helper: non-dominated sort
        def non_dominated_sort(f):
            n = len(f)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            current_front = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(f[p] <= f[q]) and np.any(f[p] < f[q]):
                        S[p].append(q)
                    elif np.all(f[q] <= f[p]) and np.any(f[q] < f[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    current_front.append(p)
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                current_front = next_front
            return fronts

        # Helper: binary tournament
        def binary_tournament(fronts, cd_cache):
            i, j = rng.integers(len(fronts[0]), size=2) if len(fronts[0]) >= 2 else (0, 0)
            # Pick from first front primarily
            front0 = fronts[0]
            i = front0[rng.integers(len(front0))]
            j = front0[rng.integers(len(front0))]
            if i == j:
                return i
            # Compare by crowding distance (higher is better)
            return i if cd_cache[i] >= cd_cache[j] else j

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Compute crowding distances for each front
            cd_cache = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    f_sub = F[front]
                    cd_sub = crowding_distance(f_sub)
                    for idx, fi in enumerate(front):
                        cd_cache[fi] = cd_sub[idx]
                elif len(front) == 2:
                    cd_cache[front[0]] = 1.0
                    cd_cache[front[1]] = 1.0
                else:
                    cd_cache[front[0]] = 1.0

            # Generate offspring
            n_offspring = pop_size
            offspring_X = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            for i in range(n_offspring):
                if time.time() > deadline:
                    break

                # Select parents
                p1 = binary_tournament(fronts, cd_cache)
                p2 = binary_tournament(fronts, cd_cache)

                # SBX crossover
                if rng.random() < 0.9:
                    child1 = np.empty(dim)
                    child2 = np.empty(dim)
                    for d in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (2 + 1.0))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1.0))
                        child1[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        child2[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                    # Mutate
                    for d in range(dim):
                        if rng.random() < 0.2:
                            child1[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                        if rng.random() < 0.2:
                            child2[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X[i] = child1
                    if i + 1 < n_offspring:
                        offspring_X[i + 1] = child2
                        offspring_F[i] = problem(child1)
                        if i + 1 < n_offspring:
                            offspring_F[i + 1] = problem(child2)
                            used += 2
                            i += 1
                        else:
                            used += 1
                    else:
                        offspring_F[i] = problem(child1)
                        used += 1
                else:
                    # Just mutate parent1
                    child = X[p1].copy()
                    for d in range(dim):
                        if rng.random() < 0.3:
                            child[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                    child = np.clip(child, lo, hi)
                    offspring_X[i] = child
                    offspring_F[i] = problem(child)
                    used += 1

            if time.time() > deadline:
                break

            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X[:n_offspring]])
            F_comb = np.vstack([F, offspring_F[:n_offspring]])

            # Select next population
            fronts_comb = non_dominated_sort(F_comb)
            next_X = np.empty((pop_size, dim))
            next_F = np.empty((pop_size, n_obj))
            count = 0
            for front in fronts_comb:
                if count + len(front) <= pop_size:
                    next_X[count:count + len(front)] = X_comb[front]
                    next_F[count:count + len(front)] = F_comb[front]
                    count += len(front)
                else:
                    # Need to truncate this front by crowding distance
                    f_sub = F_comb[front]
                    cd_sub = crowding_distance(f_sub)
                    order = np.argsort(-cd_sub)
                    needed = pop_size - count
                    selected = [front[k] for k in order[:needed]]
                    next_X[count:pop_size] = X_comb[selected]
                    next_F[count:pop_size] = F_comb[selected]
                    count = pop_size
                    break

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END
