"""Multi-objective optimizer on RE21 using DE with hypervolume-based selection."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive mutation and hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Reference-point-aligned deterministic grid initialization
        # The reference point is [3000, 0.06], so obj1 dominates HV contribution.
        # We want to cover the asymmetric space: low obj1 values are more valuable.
        n_grid = min(16, self.budget // 10)
        if n_grid < 4:
            n_grid = 4

        # Create grid points in decision space, biased toward regions that
        # might give low first objective (which dominates HV)
        grid_X = []
        # Spread across the box but with denser sampling at lower values
        for i in range(n_grid):
            # Use a biased distribution: sqrt transform to concentrate near low end
            t = (i + 0.5) / n_grid
            biased_t = np.sqrt(t)  # More points near t=0
            x = lo + biased_t * (hi - lo)
            grid_X.append(x)
        # Add a few random points for diversity
        for _ in range(max(0, n_grid - len(grid_X))):
            x = lo + (hi - lo) * rng.random(self.dim)
            grid_X.append(x)

        X = np.array(grid_X[:n_grid])
        F = np.array([problem(x) for x in X])
        used = n_grid

        # Population size for DE
        pop_size = min(30, max(10, self.budget // 8))

        # Expand population to pop_size with random points
        if used < pop_size:
            n_extra = pop_size - used
            extra_X = lo + (hi - lo) * rng.random((n_extra, self.dim))
            extra_F = np.array([problem(x) for x in extra_X])
            X = np.vstack([X, extra_X])
            F = np.vstack([F, extra_F])
            used += n_extra

        def hypervolume_2d(points, ref_pt):
            """Calculate hypervolume for 2D points dominated by reference point."""
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref_pt[1]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                if y1 < prev_y:
                    hv += (ref_pt[0] - x1) * (prev_y - y1)
                    prev_y = y1
            return hv

        def non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        def hv_contribution(point, front, ref_pt):
            """Compute HV contribution of a single point given the current front."""
            # Add point to front, compute HV, remove point, compute HV, return difference
            if len(front) == 0:
                if np.all(point < ref_pt):
                    return (ref_pt[0] - point[0]) * (ref_pt[1] - point[1])
                return 0.0

            all_pts = np.vstack([front, [point]])
            hv_with = hypervolume_2d(all_pts, ref_pt)
            hv_without = hypervolume_2d(front, ref_pt)
            return hv_with - hv_without

        # Adaptive DE parameters
        F_mut = 0.8
        CR = 0.9

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated points
            nd_mask = non_dominated_mask(F)
            n_nd = np.sum(nd_mask)
            nd_indices = np.where(nd_mask)[0]

            # If we have few non-dominated points, use all as potential parents
            if n_nd < 3:
                parent_indices = np.arange(len(F))
            else:
                parent_indices = nd_indices

            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct parents
                n_p = len(parent_indices)
                if n_p < 3:
                    parent_indices = np.arange(len(F))
                    n_p = len(parent_indices)

                idx1, idx2, idx3 = rng.choice(n_p, size=3, replace=False)
                p1, p2, p3 = parent_indices[idx1], parent_indices[idx2], parent_indices[idx3]

                # Mutation in decision space
                x1, x2, x3 = X[p1], X[p2], X[p3]
                mutant_x = x1 + F_mut * (x2 - x3)

                # Crossover
                trial = np.empty(self.dim)
                jrand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = mutant_x[j]
                    else:
                        trial[j] = X[p1][j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Selection: use hypervolume contribution for better decisions
                if np.all(f_trial <= F[p1]) and np.any(f_trial < F[p1]):
                    # Trial dominates, always replace
                    X[p1] = trial
                    F[p1] = f_trial
                elif not (np.all(F[p1] <= f_trial) and np.any(F[p1] < F[p1])):
                    # Neither dominates; compare HV contributions
                    # Build current front from non-dominated points
                    front = F[nd_mask]
                    if len(front) == 0:
                        front = F

                    # Compute HV contribution of trial vs original
                    # Remove p1 from consideration, compute front without it
                    other_indices = np.setdiff1d(np.arange(len(F)), [p1])
                    other_front = F[other_indices]
                    # Keep only non-dominated from others
                    if len(other_front) > 0:
                        other_nd = non_dominated_mask(other_front)
                        base_front = other_front[other_nd]
                    else:
                        base_front = np.empty((0, 2))

                    hv_trial = hv_contribution(f_trial, base_front, ref)
                    hv_orig = hv_contribution(F[p1], base_front, ref)

                    if hv_trial > hv_orig:
                        X[p1] = trial
                        F[p1] = f_trial

            # Adaptive mutation rate: shrink over time to exploit
            progress = used / self.budget
            F_mut = 0.9 - 0.6 * progress  # From 0.9 to 0.3
            CR = 0.9 - 0.3 * progress  # From 0.9 to 0.6

        return X
