"""Rewritten multi-objective optimizer for WFG7 with deadline guard."""
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective optimizer with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, self.budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best non-dominated set
        best_set = list(zip(X, F))

        def update_best_set(new_point):
            """Add point to best_set if not dominated, remove dominated points."""
            x, f = new_point
            # Check if new point is dominated by any existing
            for _, ef in best_set:
                if np.all(ef <= f) and np.any(ef < f):
                    return  # dominated, don't add
            # Remove any existing points dominated by new point
            best_set[:] = [(ex, ef) for ex, ef in best_set
                           if not (np.all(f <= ef) and np.any(f < ef))]
            best_set.append(new_point)

        # Initial update
        for x, f in zip(X, F):
            update_best_set((x, f))

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points with given reference point."""
            # Filter points that are better than reference in all objectives
            pts = [(f[0], f[1], f[2]) for f in points
                   if f[0] < ref[0] and f[1] < ref[1] and f[2] < ref[2]]
            if not pts:
                return 0.0

            # Sort by first objective
            pts.sort(key=lambda p: p[0])

            # Use inclusion-exclusion for 3D hypervolume
            # For each point, compute its contribution
            total_hv = 0.0
            n = len(pts)

            for i in range(n):
                x1, y1, z1 = pts[i]
                # Width in x dimension: distance to next point's x or reference
                x2 = pts[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1

                # For y-z plane, we need to compute the 2D hypervolume
                # of the "shadow" of points i..n-1 clipped to [y1, ref[1]] x [z1, ref[2]]
                # Actually, let's use a simpler approach: for each point, its contribution
                # is dx * (area in y-z where no other point dominates in both y and z)

                # Compute 2D hypervolume in y-z plane for points i..n-1
                # with reference (ref[1], ref[2])
                yz_pts = [(pts[j][1], pts[j][2]) for j in range(i, n)]
                yz_pts = [(y, z) for y, z in yz_pts if y < ref[1] and z < ref[2]]
                if not yz_pts:
                    continue

                # Sort by y
                yz_pts.sort(key=lambda p: p[0])
                m = len(yz_pts)

                # For 2D: hv = sum over sorted points of (dy * dz)
                # where dy is gap in y, dz is max z-extent not covered by previous points
                # Use sweep line: for each point in y-sorted order,
                # the contribution is dy * (ref[2] - min_z_so_far)
                # where min_z_so_far is the minimum z among points with y >= current y
                # Actually, standard 2D HV: sort by y, then for each point,
                # its z-contribution is (ref[2] - z_i) if it's the best z seen so far
                # But we need to be careful: we want the union of rectangles.

                # Simpler: sort by y, then compute the upper envelope of z
                # hv_2d = integral over y of (ref[2] - min_z(y)) dy
                # where min_z(y) = min z among points with y' >= y

                # Let's just do it directly:
                # Sort by y ascending
                # For each point j, its y-interval is [y_j, y_{j+1}] (or ref[1])
                # In that interval, the z-bound is the minimum z among points j..m-1
                # Contribution = dy * (ref[2] - min_z)

                min_z = ref[2]
                for j in range(m - 1, -1, -1):
                    y, z = yz_pts[j]
                    if z < min_z:
                        min_z = z
                    y_next = yz_pts[j + 1][0] if j + 1 < m else ref[1]
                    dy = y_next - y
                    dz = ref[2] - min_z
                    if dy > 0 and dz > 0:
                        total_hv += dx * dy * dz

            return total_hv

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated points from current population
            nondom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom_mask[i] = False
                        break

            nondom_idx = np.where(nondom_mask)[0]
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F))

            # Select two parents from non-dominated set via tournament
            p1_idx = nondom_idx[rng.integers(len(nondom_idx))]
            p2_idx = nondom_idx[rng.integers(len(nondom_idx))]
            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # Crossover: uniform
            mask = rng.random(self.dim) > 0.5
            child = np.where(mask, p1, p2)

            # Adaptive mutation: sigma decreases over time
            progress = used / self.budget
            sigma = 0.15 * (1.0 - 0.5 * progress)
            child = child + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst point in the population (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Update best set
            update_best_set((child, fc))

        # Final local search polish on non-dominated points
        if time.time() < deadline - 10:
            # Get current non-dominated points
            nondom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom_mask[i] = False
                        break

            nondom_idx = np.where(nondom_mask)[0]
            for idx in nondom_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Try small perturbations
                for _ in range(3):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    pert = X[idx] + 0.02 * (hi - lo) * rng.standard_normal(self.dim)
                    pert = np.clip(pert, lo, hi)
                    fp = problem(pert)
                    used += 1

                    # Check if perturbation is not dominated by original
                    if not (np.all(F[idx] <= fp) and np.any(F[idx] < fp)):
                        # Perturbation is not worse, try to add to best set
                        update_best_set((pert, fp))
                        # Also update population if it's better
                        if np.sum(fp) < np.sum(F[idx]):
                            X[idx] = pert
                            F[idx] = fp

        return X
