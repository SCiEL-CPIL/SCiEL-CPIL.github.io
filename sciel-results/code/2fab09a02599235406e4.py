"""Rewritten optimizer: reference-point-aware adaptive multi-objective search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive multi-objective optimizer with elitism, crowding, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Reference point for normalization (from task: [1705, 12, 0.3])
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Initial population
        pop_size = min(30, budget // 3)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Archive of best points (non-dominated, diverse)
        archive_X = list(X)
        archive_F = list(F)

        def non_dominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr (minimization)."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            # F_arr: (n, n_obj)
            # Point i dominates point j if F_i <= F_j in all objectives and F_i < F_j in at least one
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(n):
                    if i == j or is_dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        is_dominated[j] = True
            return np.where(~is_dominated)[0]

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F_arr[indices]
            for m in range(n_obj):
                vals = F_sub[:, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-15:
                    continue
                # Sort order
                order = np.argsort(vals)
                # Boundary points get inf
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        def hypervolume_contribution(point, archive_F_arr, ref_pt):
            """Approximate hypervolume contribution of a point vs archive (3D)."""
            # For 3D, contribution of point p vs a set of points is the volume of the box
            # [p_0, ref_0] x [p_1, ref_1] x [p_2, ref_2] minus overlaps.
            # Simplified: just use the box volume as a proxy (good enough for selection bias).
            box = np.prod(np.maximum(ref_pt - point, 0))
            return box

        def select_parents(archive_F_list, archive_X_list, n_parents):
            """Select parents using non-dominated sorting + crowding + hypervolume proxy."""
            F_arr = np.array(archive_F_list)
            X_arr = np.array(archive_X_list)
            nd_idx = non_dominated_indices(F_arr)
            if len(nd_idx) >= n_parents:
                # Use crowding distance to pick diverse ones
                cd = crowding_distance(F_arr, nd_idx)
                # Combine crowding with hypervolume proxy
                hv = np.array([hypervolume_contribution(F_arr[i], F_arr, ref) for i in nd_idx])
                # Normalize
                cd_norm = cd / (cd.max() + 1e-15)
                hv_norm = hv / (hv.max() + 1e-15)
                score = 0.5 * cd_norm + 0.5 * hv_norm
                # Pick top n_parents by score
                top = np.argsort(-score)[:n_parents]
                return [X_arr[nd_idx[k]] for k in top], [F_arr[nd_idx[k]] for k in top]
            else:
                # Not enough non-dominated, take all and supplement
                parents_X = [X_arr[i] for i in nd_idx]
                parents_F = [F_arr[i] for i in nd_idx]
                # Fill remaining with random from archive
                remaining = n_parents - len(parents_X)
                if remaining > 0:
                    other_idx = np.setdiff1d(np.arange(len(X_arr)), nd_idx)
                    if len(other_idx) > 0:
                        extra = rng.choice(other_idx, size=min(remaining, len(other_idx)), replace=False)
                        for i in extra:
                            parents_X.append(X_arr[i])
                            parents_F.append(F_arr[i])
                return parents_X, parents_F

        # Main loop
        gen = 0
        while used < budget:
            gen += 1
            # Adaptive mutation scale: start large, shrink
            progress = used / budget
            mut_scale = 0.3 * (1.0 - 0.7 * progress)

            # Select parents
            n_parents = max(2, min(6, len(archive_X)))
            parents_X, parents_F = select_parents(archive_F, archive_X, n_parents)

            # Generate children
            n_children = min(budget - used, 5)
            for _ in range(n_children):
                # Pick a parent
                p_idx = rng.integers(len(parents_X))
                parent = parents_X[p_idx]

                # Mutation: Gaussian with adaptive scale
                child = parent + mut_scale * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                # Occasionally do a larger jump (exploration)
                if rng.random() < 0.1:
                    child = parent + 0.5 * span * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update archive: add if non-dominated
                new_F_arr = np.array(archive_F + [fc])
                nd_idx = non_dominated_indices(new_F_arr)
                # Keep only non-dominated
                if len(nd_idx) == len(new_F_arr) - 1 or len(nd_idx) < len(archive_F):
                    # The new point is non-dominated or dominated some existing
                    archive_X = [archive_X[i] for i in nd_idx]
                    archive_F = [new_F_arr[i] for i in nd_idx]
                else:
                    # New point is dominated, but maybe we should still keep for diversity
                    # Only add if archive is small
                    if len(archive_X) < 20:
                        archive_X.append(child)
                        archive_F.append(fc)

            # Limit archive size to 50 for efficiency
            if len(archive_X) > 50:
                F_arr = np.array(archive_F)
                nd_idx = non_dominated_indices(F_arr)
                if len(nd_idx) > 50:
                    cd = crowding_distance(F_arr, nd_idx)
                    keep = np.argsort(-cd)[:50]
                    archive_X = [archive_X[nd_idx[k]] for k in keep]
                    archive_F = [F_arr[nd_idx[k]] for k in keep]
                else:
                    # Keep all non-dominated, drop dominated
                    archive_X = [archive_X[i] for i in nd_idx]
                    archive_F = [F_arr[i] for i in nd_idx]

        return archive_X[0] if archive_X else None
# EVOLVE-BLOCK-END
