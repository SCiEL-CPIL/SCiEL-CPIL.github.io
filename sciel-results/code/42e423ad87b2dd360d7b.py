# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS init, HV-guided parent selection, adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin below 330s wall-clock

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Reference point for hypervolume (used for parent scoring / selection)
        # We don't know the true ref; use a per-objective upper estimate.
        # We'll refine it as we see values.
        # Initial population size: use a chunk of the budget.
        init_size = int(min(40, max(10, self.budget // 5)))

        # Latin Hypercube sampling for initialization
        def lhs(n, d):
            idx = [rng.permutation(n) for _ in range(d)]
            u = (np.array(idx) + rng.random((d, n))) / n
            return u.T  # (n, d)

        X = lo + span * lhs(init_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = init_size

        # Track an estimated reference point: a bit above the max seen so far.
        # For parent selection we just need a consistent ref; use current max + margin.
        def hv_contrib(F, ref):
            """Compute per-point hypervolume contribution (approximate via sorting).
            For small n_obj this is fine. Returns an array of contributions."""
            n = len(F)
            if n == 0:
                return np.array([])
            # Only count points dominated by ref (better in all objectives)
            # Contribution computed by inclusion-exclusion on 3D is complex;
            # use a simpler proxy: for each point, the volume of the box it
            # uniquely covers relative to others. Approximation: sort by first
            # objective and compute incremental boxes.
            # We'll use a 3D-specific incremental HV contribution.
            # Simpler: use the 'slice' method for 3 objectives.
            # For parent selection, an approximate contribution is fine.
            # Use: contribution_i = product_j (ref_j - F_i_j) * (1/(rank-based))
            # We'll compute exact HV contribution for 3D.
            return _hv_contrib_3d(F, ref)

        def _hv_contrib_3d(F, ref):
            """Exact per-point HV contribution for 3 objectives (minimization).
            Only points better than ref in all dims contribute.
            Uses the standard 3D HV algorithm per point via 'dominated region'
            approximation: we compute the total HV of the set, and the HV of
            the set minus point i, and take the difference. For efficiency with
            small populations this is acceptable."""
            n = len(F)
            ref = np.asarray(ref, dtype=float)
            # Mask of points that are better than ref in all objectives
            valid = np.all(F < ref, axis=1)
            contrib = np.zeros(n)
            if not np.any(valid):
                return contrib
            # Total HV of valid points
            total = _hv_3d(F[valid], ref)
            for i in range(n):
                if not valid[i]:
                    contrib[i] = 0.0
                    continue
                others = np.delete(F[valid], np.where(valid)[0][i])
                if len(others) == 0:
                    contrib[i] = total
                else:
                    hv_others = _hv_3d(others, ref)
                    contrib[i] = total - hv_others
            return contrib

        def _hv_3d(F, ref):
            """Exact hypervolume for 3 objectives (minimization) with given ref.
            Only points better than ref in all objectives are considered."""
            F = np.asarray(F, dtype=float)
            ref = np.asarray(ref, dtype=float)
            if len(F) == 0:
                return 0.0
            # Filter points better than ref in all objectives
            m = np.all(F < ref, axis=1)
            F = F[m]
            if len(F) == 0:
                return 0.0
            # Sort by first objective ascending
            order = np.argsort(F[:, 0])
            F = F[order]
            # Standard 3D HV: sweep over first objective
            hv = 0.0
            # We'll use the inclusion-exclusion via 'layers'
            # For 3D, a known efficient method: sort by x, then for each point
            # compute the 2D HV of the region it dominates that is not covered
            # by previously processed (smaller x) points.
            # Process in increasing x.
            prev_y = None
            prev_z = None
            # We'll maintain the 'skyline' in y-z of points with smaller x.
            # Simpler: use the formula
            # HV = sum over i of (x_{i+1} - x_i) * HV_2D(points with x <= x_i in yz)
            # where x_{n+1} = ref[0].
            xs = np.concatenate([F[:, 0], [ref[0]]])
            # For each segment between x_i and x_{i+1}, the points active are those
            # with F[:,0] <= x_i (i.e., index < i+1... careful).
            # Let's do: for k in range(n): active points are F[0:k+1] (x <= F[k,0]).
            # segment width = xs[k+1] - xs[k]
            # 2D HV of active points in (y,z) with ref (ref[1], ref[2]).
            # But active points also must be < ref in y,z.
            for k in range(len(F)):
                active = F[:k+1]
                # 2D HV of active in yz
                hv2 = _hv_2d(active[:, 1], active[:, 2], ref[1], ref[2])
                width = xs[k+1] - xs[k]
                hv += width * hv2
            return hv

        def _hv_2d(y, z, ry, rz):
            """2D HV (minimization) for points (y,z) better than (ry,rz)."""
            pts = np.column_stack([y, z])
            m = np.all(pts < np.array([ry, rz]), axis=1)
            pts = pts[m]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            # Sweep in y; maintain max z seen (i.e., smallest z since minimization)
            # For minimization, the dominated region: for each point, the area
            # it uniquely covers.
            # Sort by y ascending. The 'front' in z is decreasing? For minimization,
            # a point with smaller y and smaller z dominates.
            # Standard: sort by y, keep track of the minimum z so far (best z).
            # Actually for HV, we want the area under the Pareto front.
            # Let's use: sort by y ascending. For each point, the z-extent it adds is
            # from its z up to the current best (smallest) z of points with smaller y?
            # No. Let me think: the dominated region is the union of rectangles
            # [y_i, ry] x [z_i, rz]. The union area.
            # Sort by y ascending. Maintain the smallest z seen so far (min_z).
            # A point contributes area (ry - y_i) * (min_z_so_far - z_i) if z_i < min_z_so_far.
            # Then update min_z_so_far = min(min_z_so_far, z_i).
            min_z = np.inf
            for i in range(len(pts)):
                yi, zi = pts[i]
                if zi < min_z:
                    hv += (ry - yi) * (min_z - zi)
                    min_z = zi
            return hv

        # Estimate reference point: use current max + a margin.
        # We'll use a ref that is slightly above the worst seen so far per objective.
        def estimate_ref(F):
            # Use max + 5% margin, but at least a small amount above min.
            mx = np.max(F, axis=0)
            mn = np.min(F, axis=0)
            span_obj = np.maximum(mx - mn, 1e-6)
            ref = mx + 0.05 * span_obj
            return ref

        ref = estimate_ref(F)

        # Main loop
        step = 0
        while used < self.budget:
            if time.time() > deadline:
                break

            # Update reference estimate
            ref = estimate_ref(F)

            # Compute HV contributions for parent selection
            contrib = hv_contrib(F, ref)
            # Elite = point with max contribution
            # Pick a parent from the top contributors (with some randomness)
            n_parents = min(5, len(contrib))
            top_idx = np.argsort(-contrib)[:n_parents]
            parent_idx = top_idx[rng.integers(n_parents)]
            parent = X[parent_idx]

            # Adaptive mutation: step size shrinks with progress
            progress = used / self.budget
            base_step = 0.3 * (1.0 - 0.5 * progress)
            # Occasionally do a larger exploratory jump
            if rng.random() < 0.1:
                scale = base_step * 3.0
            else:
                scale = base_step
            child = parent + scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the point with the smallest HV contribution
            worst_idx = int(np.argmin(contrib))
            X[worst_idx] = child
            F[worst_idx] = fc
            step += 1

        return X
# EVOLVE-BLOCK-END
