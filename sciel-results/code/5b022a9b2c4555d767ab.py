"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        rng = np.random.RandomState(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all non-dominated points found (for hypervolume)
        all_points = list(F)

        # ZDT3 specific: x1 is the first variable, others are auxiliary
        # For ZDT3, g(x) depends on x2..x30, f1 = x1, f2 = g*(1 - sqrt(f1/g))
        # The Pareto front is discontinuous. We need to spread x1 and minimize g.

        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            def non_dominated_sort(F):
                n = len(F)
                S = [[] for _ in range(n)]
                n_count = np.zeros(n, dtype=int)
                fronts = [[]]

                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        # i dominates j if i is better in all and strictly better in at least one
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            n_count[i] += 1

                for i in range(n):
                    if n_count[i] == 0:
                        fronts[0].append(i)

                k = 0
                while len(fronts[k]) > 0:
                    Q = []
                    for i in fronts[k]:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                Q.append(j)
                    k += 1
                    if Q:
                        fronts.append(Q)
                    else:
                        break
                return fronts

            fronts = non_dominated_sort(F)

            # Crowding distance
            def crowding_distance(F, front_indices):
                if len(front_indices) <= 2:
                    return np.ones(len(front_indices))
                m = len(front_indices)
                cd = np.zeros(m)
                for k in range(n_obj):
                    idx = np.argsort(F[front_indices, k])
                    cd[idx[0]] = np.inf
                    cd[idx[-1]] = np.inf
                    f_range = F[front_indices, k].max() - F[front_indices, k].min()
                    if f_range == 0:
                        continue
                    for i in range(1, m - 1):
                        cd[idx[i]] += (F[front_indices[idx[i+1]], k] - F[front_indices[idx[i-1]], k]) / f_range
                return cd

            # Selection: binary tournament
            def tournament_select(fronts, cd_map, F):
                def rank_of(i):
                    for r, front in enumerate(fronts):
                        if i in front:
                            return r
                    return len(fronts)
                a = rng.randint(len(F))
                b = rng.randint(len(F))
                ra, rb = rank_of(a), rank_of(b)
                if ra != rb:
                    return a if ra < rb else b
                # Same rank, use crowding distance
                ca = cd_map.get(a, 0)
                cb = cd_map.get(b, 0)
                return a if ca > cb else b

            # Build crowding distance map
            cd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cds = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_map[idx] = cds[i]

            # Generate children
            n_children = min(pop_size, budget - used)
            if n_children <= 0:
                break

            child_X = []
            child_F = []
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                p1 = tournament_select(fronts, cd_map, F)
                p2 = tournament_select(fronts, cd_map, F)

                # SBX crossover
                child = np.empty(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        # Crossover
                        if rng.random() < 0.5:
                            child[d] = X[p1, d]
                        else:
                            child[d] = X[p2, d]
                    else:
                        # No crossover, just pick one
                        if rng.random() < 0.5:
                            child[d] = X[p1, d]
                        else:
                            child[d] = X[p2, d]

                # Mutation
                # For ZDT3, we want to explore x1 well and minimize g
                # Use polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        # Polynomial mutation
                        delta = rng.random()
                        if delta < 0.5:
                            mut_pow = 20.0
                            delta = (2 * delta) ** (1 / (mut_pow + 1)) - 1
                        else:
                            mut_pow = 20.0
                            delta = 1 - (2 * (1 - delta)) ** (1 / (mut_pow + 1))
                        child[d] += delta * (hi[d] - lo[d])
                        child[d] = np.clip(child[d], lo[d], hi[d])

                fc = problem(child)
                child_X.append(child)
                child_F.append(fc)
                used += 1
                all_points.append(fc)

            if not child_X:
                break

            child_X = np.array(child_X)
            child_F = np.array(child_F)

            # Combine parent and child
            X_comb = np.vstack([X, child_X])
            F_comb = np.vstack([F, child_F])

            # Non-dominated sorting on combined
            fronts_comb = non_dominated_sort(F_comb)

            # Select next generation
            X_next = []
            F_next = []
            for front in fronts_comb:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(X_next)
                    cds = crowding_distance(F_comb, front)
                    order = np.argsort(cds)[::-1]  # descending crowding distance
                    for i in order[:remaining]:
                        idx = front[i]
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break

            X = np.array(X_next)
            F = np.array(F_next)

        return X
# EVOLVE-BLOCK-END
