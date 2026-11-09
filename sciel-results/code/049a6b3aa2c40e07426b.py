# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, max(20, budget // 4))
        # Cross-over rate
        crossover_rate = 0.9
        # Mutation rate (per gene)
        mutation_rate = 1.0 / dim
        # Polynomial mutation exponent
        eta = 20.0

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    fronts.append([p])
            i = 0
            while i < len(fronts):
                current_front = fronts[i]
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1
            return fronts

        def crowding_distance(F, indices):
            """Returns crowding distance array for the given indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            distances = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort([F[i, m] for i in indices])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_min = F[indices[sorted_idx[0]], m]
                f_max = F[indices[sorted_idx[-1]], m]
                if f_max - f_min < 1e-15:
                    continue
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (
                        F[indices[sorted_idx[k + 1]], m] - F[indices[sorted_idx[k - 1]], m]
                    ) / (f_max - f_min)
            return distances

        def tournament_select(X, F, front_ids, crowd_dists):
            """Tournament selection of two parents."""
            def rank_and_dist(idx):
                return front_ids[idx], crowd_dists[idx]

            def better(a, b):
                ra, da = rank_and_dist(a)
                rb, db = rank_and_dist(b)
                if ra != rb:
                    return ra < rb
                return da > db

            idx1 = rng.integers(len(X))
            idx2 = rng.integers(len(X))
            p1 = idx1 if better(idx1, idx2) else idx2
            idx1 = rng.integers(len(X))
            idx2 = rng.integers(len(X))
            p2 = idx1 if better(idx1, idx2) else idx2
            return p1, p2

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() > crossover_rate:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        continue
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        def polynomial_mutate(x):
            """Polynomial mutation."""
            child = x.copy()
            for i in range(dim):
                if rng.random() < mutation_rate:
                    u = rng.random()
                    delta1 = (1 + 2 * (x[i] - lo[i]) / (hi[i] - lo[i])) ** (eta + 1)
                    delta2 = (1 + 2 * (hi[i] - x[i]) / (hi[i] - lo[i])) ** (eta + 1)
                    if u < 0.5:
                        deltaq = (1 / delta1) * (2 * u - 1) + 1
                    else:
                        deltaq = 1 - (1 / delta2) * (2 * (1 - u) - 1)
                    child[i] = x[i] + (hi[i] - lo[i]) * deltaq
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop
        while used < budget:
            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)
            front_ids = np.zeros(len(X), dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    front_ids[idx] = front_idx

            # Crowding distance for each front
            crowd_dists = np.zeros(len(X))
            for front in fronts:
                cd = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    crowd_dists[idx] = cd[j]

            # Create offspring
            n_offspring = min(pop_size, budget - used)
            children = []
            for _ in range(n_offspring):
                p1, p2 = tournament_select(X, F, front_ids, crowd_dists)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                children.append(c1)
                children.append(c2)
                used += 2
                if used >= budget:
                    break

            # Evaluate children
            children = np.array(children[:min(len(children), budget - used + pop_size)])
            # Trim to fit budget
            max_children = budget - used
            if len(children) > max_children:
                children = children[:max_children]
                used += len(children)
            else:
                used += len(children)

            F_children = np.array([problem(x) for x in children])

            # Combine parent and offspring
            X_combined = np.vstack([X, children])
            F_combined = np.vstack([F, F_children])

            # Sort combined population
            fronts_combined = fast_non_dominated_sort(F_combined)
            front_ids_combined = np.zeros(len(X_combined), dtype=int)
            for front_idx, front in enumerate(fronts_combined):
                for idx in front:
                    front_ids_combined[idx] = front_idx

            crowd_dists_combined = np.zeros(len(X_combined))
            for front in fronts_combined:
                cd = crowding_distance(F_combined, front)
                for j, idx in enumerate(front):
                    crowd_dists_combined[idx] = cd[j]

            # Select next population
            new_X = []
            new_F = []
            for front in fronts_combined:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    remaining = pop_size - len(new_X)
                    # Sort front by crowding distance
                    front_sorted = sorted(front, key=lambda i: crowd_dists_combined[i], reverse=True)
                    for idx in front_sorted[:remaining]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
