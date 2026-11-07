"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for WFG7 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(10, self.budget // 4))

        # Initial population
        X = lo + rng_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume-like dominance checks (not needed for algorithm, but good for pruning)
        # We'll use a simple non-dominated sort approach

        def non_dominated_sort(F):
            """Return non-dominated indices."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points."""
            n_nd = len(nd_indices)
            if n_nd <= 2:
                return np.ones(n_nd)
            distances = np.zeros(n_nd)
            F_nd = F[nd_indices]
            for m in range(self.n_obj):
                min_val = F_nd[:, m].min()
                max_val = F_nd[:, m].max()
                if max_val - min_val < 1e-15:
                    continue
                idx_min = np.argmin(F_nd[:, m])
                idx_max = np.argmax(F_nd[:, m])
                distances[idx_min] = np.inf
                distances[idx_max] = np.inf
                for i in range(n_nd):
                    if i == idx_min or i == idx_max:
                        continue
                    # Find neighbors
                    neighbors = np.where((F_nd[:, m] > F_nd[i, m]) & (F_nd[:, m] < F_nd[i, m]))
                    if len(neighbors) >= 2:
                        neighbors = np.sort(neighbors)
                        for k in range(len(neighbors) - 1):
                            if neighbors[k + 1] == i:
                                distances[i] += (F_nd[neighbors[k + 1], m] - F_nd[neighbors[k], m]) / (max_val - min_val)
                    elif len(neighbors) == 1:
                        distances[i] += (max_val - min_val) / (max_val - min_val)
            return distances

        while used < self.budget:
            # Check time
            if time.time() > deadline - 2.0:
                break

            # Find non-dominated set
            nd_idx = non_dominated_sort(F)

            if len(nd_idx) == 0:
                # All dominated, pick random
                nd_idx = np.array([rng.integers(pop_size)])

            # Compute crowding distance for non-dominated
            cd = crowding_distance(F, nd_idx)

            # Selection: pick parent with high crowding distance
            # Use tournament-like selection
            n_tournament = min(5, len(nd_idx))
            tournament = rng.choice(len(nd_idx), size=n_tournament, replace=False)
            best_t = tournament[np.argmax(cd[tournament])]
            parent_idx = nd_idx[best_t]
            parent = X[parent_idx]

            # Mutation: adaptive step size
            # Use a mix of global and local search
            if rng.random() < 0.3:
                # Global restart
                step = 0.5
            else:
                # Local refinement
                step = 0.1 + 0.2 * (used / self.budget)

            mutation = step * rng_ * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: remove the worst dominated point or the point with lowest crowding distance among dominated
            # Find dominated points
            nd_idx_new = non_dominated_sort(F)
            dominated_idx = np.setdiff1d(np.arange(pop_size), nd_idx_new)

            if len(dominated_idx) > 0:
                # Replace the most dominated (worst) point
                # Score by how many points dominate it
                worst_score = -np.inf
                worst_idx = dominated_idx[0]
                for di in dominated_idx:
                    n_dom = 0
                    for fj in F:
                        if np.all(fj <= F[di]) and np.any(fj < F[di]):
                            n_dom += 1
                    if n_dom > worst_score:
                        worst_score = n_dom
                        worst_idx = di
                X[worst_idx] = child
                F[worst_idx] = fc
            else:
                # All non-dominated, replace the one with lowest crowding distance
                cd_all = crowding_distance(F, np.arange(pop_size))
                worst_idx = np.argmin(cd_all)
                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
