"""Rewritten multi-objective optimizer for WFG4 with deadline guard."""
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
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        pop_size = 40
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (all <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination relation
            n_dominates = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if dominates(F[i], F[j]):
                            S[i].append(j)
                        elif dominates(F[j], F[i]):
                            n_dominates[i] += 1
                if n_dominates[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-10:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)
            return cd

        def tournament_select(F, X, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            fronts = non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Compute crowding distance
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd_vals = crowding_distance(F, np.array(front))
                    for i, idx in enumerate(front):
                        cd[idx] = cd_vals[i]
            # Tournament
            candidates = rng.integers(0, len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                    best = c
            return X[best].copy()

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            return child1, child2

        def mutate(x, rate=0.1):
            """Gaussian mutation with adaptive step size."""
            x_new = x.copy()
            mask = rng.random(self.dim) < rate
            step = 0.1 * (hi - lo)
            x_new[mask] += step[mask] * rng.standard_normal(np.sum(mask))
            return np.clip(x_new, lo, hi)

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            children = []
            for _ in range(pop_size // 2):
                if time.time() > deadline:
                    break
                p1 = tournament_select(F, X)
                p2 = tournament_select(F, X)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                children.append(c1)
                children.append(c2)

            if not children:
                break

            # Evaluate children
            F_child = np.array([problem(c) for c in children])
            used += len(children)

            # Combine parent and child populations
            X_combined = np.vstack([X, np.array(children)])
            F_combined = np.vstack([F, F_child])

            # Non-dominated sort combined population
            fronts = non_dominated_sort(F_combined)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    remaining = pop_size - len(next_X)
                    # Sort front by crowding distance
                    front_arr = np.array(front)
                    cd_vals = crowding_distance(F_combined, front_arr)
                    order = np.argsort(-cd_vals)
                    for i in range(remaining):
                        idx = front_arr[order[i]]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
