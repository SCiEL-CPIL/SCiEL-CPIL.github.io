"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-style selection with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []

        # Initial population
        pop_size = min(50, budget // 4)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        all_X.extend(X.tolist())
        all_F.extend(F.tolist())

        def non_dominated_mask(F_mat):
            """Return boolean mask of non-dominated points in F_mat."""
            n = len(F_mat)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F_mat - F_mat[i]
                dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = False
            return mask

        def crowding_distance(F_mat, mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F_mat)
            dist = np.zeros(n)
            if not np.any(mask) or np.sum(mask) <= 2:
                return dist
            idx = np.where(mask)[0]
            for m in range(n_obj):
                order = idx[np.argsort(F_mat[idx, m])]
                if len(order) > 2:
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    f_range = F_mat[order[-1], m] - F_mat[order[0], m]
                    if f_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            dist[order[i]] += (F_mat[order[i + 1], m] - F_mat[order[i - 1], m]) / f_range
            return dist

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_mask = non_dominated_mask(F)
            nd_count = np.sum(nd_mask)

            if nd_count == 0:
                parents_idx = np.arange(len(F))
            else:
                parents_idx = np.where(nd_mask)[0]

            # Select parent via tournament on non-dominated + crowding
            if len(parents_idx) > 1:
                cd = crowding_distance(F, nd_mask)
                c1, c2 = np.random.choice(parents_idx, 2, replace=False)
                if cd[c1] >= cd[c2]:
                    parent = X[c1]
                else:
                    parent = X[c2]
            else:
                parent = X[parents_idx[0]]

            # Adaptive mutation: start large, shrink over time
            progress = used / budget
            sigma = (1.0 - 0.7 * progress) * 0.3 * (hi - lo)
            child = parent + sigma * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            all_X.append(child)
            all_F.append(fc)

            # Replace worst point (highest sum of objectives among dominated)
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                dom_idx = np.where(dom_mask)[0]
                worst_idx = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
            else:
                cd = crowding_distance(F, nd_mask)
                worst_idx = np.argmin(cd)

            X[worst_idx] = child
            F[worst_idx] = fc

        return all_X
# EVOLVE-BLOCK-END
