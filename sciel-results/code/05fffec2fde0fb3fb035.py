"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining NSGA-II-style selection with hypervolume-guided parent choice."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        # Initial population
        pop_size = min(40, max(10, budget // 5))
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F - F[i]
                dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = False
            return mask

        def crowding_distance(F, mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F)
            dist = np.zeros(n)
            if not np.any(mask) or np.sum(mask) <= 2:
                return dist
            idx = np.where(mask)[0]
            for m in range(n_obj):
                order = idx[np.argsort(F[idx, m])]
                if len(order) > 2:
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    f_range = F[order[-1], m] - F[order[0], m]
                    if f_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
            return dist

        def hv_contribution(f, others, ref):
            """Compute hypervolume contribution of point f given others (all within ref)."""
            # Filter others that are better than ref
            others_valid = others[np.all(others < ref, axis=1)]
            if len(others_valid) == 0:
                # Hypervolume of a single point vs reference in 3D
                if np.any(f >= ref):
                    return 0.0
                return (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            # For simplicity in 3D, approximate contribution by checking if f is dominated
            # by any other point; if so, contribution is 0
            for o in others_valid:
                if np.all(o <= f) and np.any(o < f):
                    return 0.0
            # Otherwise approximate as full box minus overlaps (crude but fast)
            # Use a simple 1D projection heuristic for speed
            return (ref[0] - max(f[0], np.max(others_valid[:, 0]))) * \
                   (ref[1] - max(f[1], np.max(others_valid[:, 1]))) * \
                   (ref[2] - max(f[2], np.max(others_valid[:, 2])))

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

            # Hypervolume-guided parent selection
            if len(parents_idx) > 1:
                # Compute approximate HV contribution for each non-dominated point
                cd = crowding_distance(F, nd_mask)
                # Use crowding as a proxy for HV contribution (faster)
                # Pick 2 random candidates, choose one with higher crowding
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

            # Replace worst point
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                dom_idx = np.where(dom_mask)[0]
                worst_idx = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
            else:
                cd = crowding_distance(F, nd_mask)
                worst_idx = np.argmin(cd)

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
