"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using NSGA-II style search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, max(20, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            nd = np.zeros(n, dtype=int)  # domination counts
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for idx in fronts[i]:
                    for j in S[idx]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            Q.append(j)
                if Q:
                    i += 1
                    fronts.append(Q)
                else:
                    break

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            cd = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        def select_parent(fronts, F, X, rng):
            """Select a parent using tournament on rank + crowding."""
            # Build rank array
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances per front
            crowd = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # Tournament of 2
            i1 = rng.integers(len(F))
            i2 = rng.integers(len(F))
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                return i1 if crowd[i1] > crowd[i2] else i2

        # Main evolution loop
        while used < budget:
            # Time check
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parent
            parent_idx = select_parent(fronts, F, X, rng)
            parent = X[parent_idx]

            # Adaptive mutation
            progress = used / budget
            mut_scale = 0.12 * (1.0 - 0.6 * progress)
            mutation = mut_scale * (hi - lo) * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst: highest rank, then lowest crowding
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            max_rank = ranks.max()
            worst_candidates = np.where(ranks == max_rank)[0]
            if len(worst_candidates) > 1:
                cd = crowding_distance(F, worst_candidates)
                worst_idx = worst_candidates[np.argmin(cd)]
            else:
                worst_idx = worst_candidates[0]

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
