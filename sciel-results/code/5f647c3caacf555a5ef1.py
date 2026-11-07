"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """HV-guided multi-objective search: LHS init, hypervolume parent selection, adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs(self, n, dim, lo, hi):
        """Latin hypercube sampling for good initial spread."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            u = (perm + np.random.random(n)) / n
            pts[:, d] = lo[d] + (hi[d] - lo[d]) * u
        return pts

    def _hv_contribution(self, f, ref):
        """Exact 3D hypervolume contribution of point f given the set of points (including itself)."""
        # Only consider points strictly better than ref in all objectives
        valid = np.all(f < ref, axis=1)
        if not np.any(valid):
            return 0.0
        pts = f[valid]
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # For each point, compute the box it uniquely covers
        total = 0.0
        for i in range(len(pts)):
            # width in obj0: from pts[i,0] to next point's obj0 (or ref)
            next0 = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
            w0 = ref[0] - pts[i, 0]
            # For obj1 and obj2: the region not covered by any later (larger obj0) point
            # that also has obj1 <= pts[i,1] and obj2 <= pts[i,2]
            later = pts[i + 1:]
            if len(later) == 0:
                w1 = ref[1] - pts[i, 1]
                w2 = ref[2] - pts[i, 2]
            else:
                # Among later points, find those that cover part of this box
                # w1: the max obj1 among later points with obj1 < pts[i,1]... 
                # Actually the contribution in 3D is more complex. Use inclusion via sorting.
                # Simpler: for each point, its contribution is the volume of the box
                # [pts[i,0], next0) x [pts[i,1], ...) x [pts[i,2], ...) that is not covered.
                # This exact per-point contribution requires 2D hypervolume of the "shadow".
                # Use the standard approach: contribution = w0 * (2D HV of the remaining region).
                # For efficiency and correctness, compute 2D HV of points dominated by this point's (obj1,obj2)
                # but restricted.
                # Let's use a cleaner exact method below instead.
                pass
        # Fallback: use a robust exact 3D HV computation (sweep).
        return self._hv_3d(f, ref)

    def _hv_3d(self, f, ref):
        """Exact 3D hypervolume for a set of points vs ref."""
        valid = np.all(f < ref, axis=1)
        pts = f[valid]
        if len(pts) == 0:
            return 0.0
        # Remove points dominated by another (keep non-dominated only for HV, dominated don't add)
        pts = self._non_dominated(pts)
        # Sort by obj0 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        total = 0.0
        n = len(pts)
        for i in range(n):
            # width in obj0
            next0 = pts[i + 1, 0] if i + 1 < n else ref[0]
            w0 = next0 - pts[i, 0]
            if w0 <= 0:
                continue
            # 2D hypervolume of the "shadow": points j>i with obj1<=pts[i,1] and obj2<=pts[i,2]
            # The area covered in (obj1,obj2) plane by this point's box minus what later points cover.
            # The contribution in 2D: the rectangle [pts[i,1], ref1]x[pts[i,2], ref2]
            # minus the union of rectangles from later points that lie within it.
            # Compute 2D HV of later points clipped to this rectangle.
            later = pts[i + 1:]
            if len(later) == 0:
                area = (ref[1] - pts[i, 1]) * (ref[2] - pts[i, 2])
            else:
                # clip later points to the rectangle
                cand = later[(later[:, 1] <= pts[i, 1]) & (later[:, 2] <= pts[i, 2])]
                if len(cand) == 0:
                    area = (ref[1] - pts[i, 1]) * (ref[2] - pts[i, 2])
                else:
                    # 2D HV of cand within [pts[i,1],ref1]x[pts[i,2],ref2]
                    area = self._hv_2d_in_box(cand, pts[i, 1], pts[i, 2], ref[1], ref[2])
            total += w0 * area
        return total

    def _hv_2d_in_box(self, cand, lo1, lo2, hi1, hi2):
        """2D HV of cand points within box [lo1,hi1]x[lo2,hi2], where cand points are lower-left corners."""
        # cand points have obj1, obj2; they cover [obj1,hi1]x[obj2,hi2]
        # Compute area of union of these rectangles.
        if len(cand) == 0:
            return 0.0
        # Sort by obj1
        order = np.argsort(cand[:, 0])
        cand = cand[order]
        # Sweep: maintain max obj2 covered so far
        total_area = 0.0
        prev_x = cand[0, 0]
        # We want area of union of rectangles [x_i, hi1] x [y_i, hi2]
        # Equivalent to hi1*hi2 - area of complement. Easier: sweep.
        # The union area: integrate over x from min to hi1 the max y-extent.
        # For x in [cand[k,0], cand[k+1,0]), the covered y is [min y among active, hi2]? 
        # Actually each rectangle covers y from y_i to hi2. Union over active = [min active y_i, hi2].
        # Active set = points with x_i <= current x. Since sorted by x, all points up to index are active.
        # The min y among active points determines the bottom of the union (since all extend to hi2).
        n = len(cand)
        # Build segments
        xs = list(cand[:, 0]) + [hi1]
        # For each segment [xs[k], xs[k+1]), active = points with x_i <= xs[k] (i.e., index <= k)
        min_y = hi2
        for k in range(n):
            # segment from xs[k] to xs[k+1]
            # active points are indices 0..k
            # min y among 0..k
            if cand[k, 1] < min_y:
                min_y = cand[k, 1]
            seg_w = xs[k + 1] - xs[k]
            if seg_w > 0 and min_y < hi2:
                total_area += seg_w * (hi2 - min_y)
        return total_area

    def _non_dominated(self, F):
        """Return indices-free array of non-dominated points."""
        n = len(F)
        if n == 0:
            return F
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    keep[i] = False
                    break
        return F[keep]

    def _hv_of_set(self, F, ref):
        """Hypervolume of a set of points (for parent selection)."""
        return self._hv_3d(F, ref)

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial population via LHS
        pop_size = min(30, max(10, self.budget // 6))
        X = self._lhs(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume of the current population
        # We'll do HV-guided parent selection: pick parent with highest marginal HV contribution,
        # then mutate.

        # Precompute non-dominated set for efficiency
        def non_dominated_idx(F):
            n = len(F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        keep[i] = False
                        break
            return np.where(keep)[0]

        # Adaptive mutation scale
        mut_scale = 0.15
        min_scale = 0.02
        max_scale = 0.4

        while used < self.budget:
            # Time guard
            if time.time() > deadline - 5.0:
                break

            # Select parent: among non-dominated, choose one with highest marginal HV contribution
            nd_idx = non_dominated_idx(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Compute marginal HV contribution for each non-dominated point
            base_hv = self._hv_of_set(F[nd_idx], ref)
            best_parent = nd_idx[0]
            best_contrib = -1.0
            for idx in nd_idx:
                others = np.delete(F[nd_idx], list(nd_idx).index(idx))
                hv_without = self._hv_of_set(others, ref) if len(others) > 0 else 0.0
                contrib = base_hv - hv_without
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_parent = idx

            parent = X[best_parent]

            # Adaptive mutation: Gaussian with adaptive scale
            sigma = mut_scale * (hi - lo)
            child = parent + rng.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace: if child is non-dominated, replace the point with lowest marginal HV
            # Else replace the worst (dominated) point
            # Compute marginal HV for all points to find the worst
            nd_idx_all = non_dominated_idx(F)
            if len(nd_idx_all) < len(F):
                # There are dominated points; replace one of them
                dominated_idx = np.setdiff1d(np.arange(len(F)), nd_idx_all)
                # Among dominated, pick the one that is most dominated (highest sum)
                worst = dominated_idx[np.argmax(np.sum(F[dominated_idx], axis=1))]
                X[worst] = child
                F[worst] = fc
            else:
                # All non-dominated; replace the one with lowest marginal HV contribution
                base_hv_all = self._hv_of_set(F, ref)
                worst = 0
                worst_contrib = np.inf
                for i in range(len(F)):
                    others = np.delete(F, i)
                    hv_without = self._hv_of_set(others, ref) if len(others) > 0 else 0.0
                    contrib = base_hv_all - hv_without
                    if contrib < worst_contrib:
                        worst_contrib = contrib
                        worst = i
                X[worst] = child
                F[worst] = fc

            # Adapt mutation scale: if child improved HV, keep; else reduce
            new_hv = self._hv_of_set(F, ref)
            if new_hv > base_hv_all if 'base_hv_all' in dir() else True:
                mut_scale = min(max_scale, mut_scale * 1.05)
            else:
                mut_scale = max(min_scale, mut_scale * 0.95)

        return X
# EVOLVE-BLOCK-END
