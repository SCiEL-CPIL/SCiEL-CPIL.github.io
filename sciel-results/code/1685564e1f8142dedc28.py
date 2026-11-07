"""Multi-objective optimizer for WFG7 with deadline guard."""
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
        span = hi - lo
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, self.budget // 5))

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
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
            total_hv = 0.0
            n = len(pts)

            for i in range(n):
                x1, y1, z1 = pts[i]
                # Width in x dimension: distance to next point's x or reference
                x2 = pts[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1

                # For y-z plane, compute the 2D hypervolume
                yz_pts = [(pts[j][1], pts[j][2]) for j in range(i, n)]
                yz_pts = [(y, z) for y, z in yz_pts if y < ref[1] and z < ref[2]]
                if not yz_pts:
                    continue

                # Sort by y
                yz_pts.sort(key=lambda p: p[0])
                m = len(yz_pts)

                # For 2D: sweep line from back to front
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

        def hv_contribution(f, archive_F, ref):
            """Compute approximate hypervolume contribution of point f."""
            # Check if f is better than reference
            if np.any(f >= ref):
                return 0.0

            # Check if dominated by any existing point
            for pf in archive_F:
                if np.all(pf <= f) and np.any(pf < f):
                    return 0.0

            # Compute the volume of the box dominated by f
            box_vol = np.prod(ref - f)

            # Subtract overlaps with existing points (approximate)
            # For each existing point, compute the overlap volume
            # and subtract it (this is an upper bound, but works for ranking)
            for pf in archive_F:
                # Overlap is the intersection of f's box and pf's box
                # f's box: [f0, ref0] x [f1, ref1] x [f2, ref2]
                # pf's box: [pf0, ref0] x [pf1, ref1] x [pf2, ref2]
                # Intersection: [max(f0,pf0), ref0] x [max(f1,pf1), ref1] x [max(f2,pf2), ref2]
                overlap = np.prod(np.maximum(ref - np.maximum(f, pf), 0))
                box_vol -= overlap

            return max(0, box_vol)

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
            # Use hypervolume contribution as fitness
            ref = np.array([3.0, 5.0, 7.0])
            archive_F = [f for _, f in best_set]
            contributions = np.array([hv_contribution(F[idx], archive_F, ref) 
                                     for idx in nondom_idx])
            
            # Normalize contributions for tournament
            max_contrib = np.max(contributions) if len(contributions) > 0 else 1
            if max_contrib > 0:
                contributions = contributions / max_contrib
            
            # Tournament selection: pick 2, choose better
            if len(nondom_idx) >= 2:
                t1 = rng.integers(len(nondom_idx))
                t2 = rng.integers(len(nondom_idx))
                if contributions[t1] >= contributions[t2]:
                    p1_idx = nondom_idx[t1]
                else:
                    p1_idx = nondom_idx[t2]
                
                t3 = rng.integers(len(nondom_idx))
                t4 = rng.integers(len(nondom_idx))
                if contributions[t3] >= contributions[t4]:
                    p2_idx = nondom_idx[t3]
                else:
                    p2_idx = nondom_idx[t4]
            else:
                p1_idx = nondom_idx[0]
                p2_idx = nondom_idx[0]

            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # Crossover: uniform
            mask = rng.random(self.dim) > 0.5
            child = np.where(mask, p1, p2)

            # Adaptive mutation: sigma decreases over time
            progress = used / self.budget
            sigma = 0.15 * (1.0 - 0.5 * progress)
            child = child + sigma * span * rng.standard_normal(self.dim)
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

                    pert = X[idx] + 0.02 * span * rng.standard_normal(self.dim)
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
