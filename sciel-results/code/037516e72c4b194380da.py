"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-driven sequential search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv3(self, F, ref):
        """Compute exact hypervolume for 3 objectives. F is (n, 3), ref is (3,)."""
        n = len(F)
        if n == 0:
            return 0.0
        
        # Filter points strictly better than reference
        valid = np.all(F < ref, axis=1)
        if not np.any(valid):
            return 0.0
        
        Fv = F[valid]
        n_valid = len(Fv)
        
        if n_valid == 1:
            return float(np.prod(ref - Fv[0]))
        
        # Sort by first objective
        idx = np.argsort(Fv[:, 0])
        Fv = Fv[idx]
        
        # For each point i, the contribution is the hypervolume in 2D of the region
        # that is dominated by point i but not by any point j with Fv[j,0] < Fv[i,0]
        # In 3D: for point i, the x-range is [Fv[i,0], next_x) where next_x is Fv[i+1,0] or ref[0]
        # The y-z region must be dominated by Fv[i] but not by any earlier point in y-z space
        
        # More efficient: sweep from best to worst in objective 0
        # For each point i, the unique contribution is:
        #   (x_i_width) * (2D hypervolume of (y_i, z_i) relative to the "upper envelope" of previous points)
        
        # Let's use a different approach: 
        # Sort by obj0 ascending. For each point i, its contribution is:
        #   width_x * hv2d_of_point_i_given_previous_points
        # where width_x = min(Fv[i+1,0] or ref[0]) - Fv[i,0]
        # and hv2d is the 2D hypervolume that point i contributes given the set of points that have obj0 <= Fv[i,0]
        
        # Actually, let's use the standard 3D HV algorithm:
        # Sort by first objective. Maintain a "skyline" in the 2D (obj1, obj2) space.
        
        hv = 0.0
        # We'll process points from worst obj0 to best obj0 (reverse order)
        # No wait, standard approach: sort by obj0 ascending, then for each point,
        # the contribution is the volume of the box [Fv[i], ref) minus the volume already covered by points with better obj0.
        
        # Let me use a cleaner algorithm:
        # Sort by obj0 ascending
        # For i from 0 to n-1:
        #   x_width = (Fv[i+1,0] if i < n-1 else ref[0]) - Fv[i,0]
        #   The 2D region in (obj1, obj2) that point i uniquely covers is the region
        #   dominated by (Fv[i,1], Fv[i,2]) but not dominated by any (Fv[j,1], Fv[j,2]) for j < i
        #   (since j < i means Fv[j,0] <= Fv[i,0], so those points have better or equal obj0)
        
        # This requires maintaining a 2D skyline which is complex. Let me use a simpler O(n^2) approach.
        
        # O(n^2) approach: for each point, compute the box it dominates, then subtract
        # the parts covered by other points. But this is tricky.
        
        # Let me just use a simple but correct O(n^3) or O(n^2) method for small n.
        
        # Actually, for the archive-based approach, n will be small (< 50), so let's use
        # a simple inclusion-exclusion or direct computation.
        
        # Simple approach: use the fact that for 3D, we can sort by obj0, then for each
        # consecutive pair in obj0, compute the 2D HV in the slab.
        
        # Even simpler: just compute the union of boxes directly using a grid-free method.
        
        # Let me use a recursive approach or just a straightforward one.
        
        # For small n, let's just compute it directly:
        # The hypervolume is the volume of the union of boxes [Fv[i], ref) for all i.
        
        # Use the inclusion-exclusion is too complex. Let me use a sweep line.
        
        # Actually, the cleanest for 3D with small n:
        # Sort by obj0. For each i, the contribution is:
        #   dx_i * (2D HV of point i in the region not covered by previous points in obj1-obj2)
        
        # Let me just implement a simple but correct method:
        
        # Method: for each point i, compute the box [Fv[i], ref). The hypervolume is the
        # volume of the union of all such boxes. We can compute this by:
        # 1. Sort points by obj0
        # 2. For each point i (in order of increasing obj0), the new volume added is:
        #    (next_obj0 - Fv[i,0]) * (2D area of the region in (obj1,obj2) that is
        #    below (Fv[i,1], Fv[i,2]) and not covered by the "skyline" of previous points)
        
        # The 2D skyline of previous points: a set of non-dominated points in (obj1, obj2).
        # The area under the skyline that is also below (Fv[i,1], Fv[i,2]) is what we subtract.
        
        # This is getting complex. Let me just use a simple O(n^3) method.
        
        # O(n^3): For each point i, its unique contribution is the volume of [Fv[i], ref)
        # minus the volume of [Fv[i], ref) intersected with the union of [Fv[j], ref) for j != i
        # where Fv[j] dominates Fv[i] in at least one way... no, this doesn't work simply.
        
        # Let me just use a different approach: since n is small, use the fact that
        # hypervolume in 3D can be computed by sorting and using a 2D HV subroutine.
        
        # I'll implement a proper 3D HV algorithm:
        
        # Sort by first objective
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        
        # Now Fv is sorted by obj0 ascending
        # For each i, the x-interval is [Fv[i,0], Fv[i+1,0]) for i < n-1, and [Fv[n-1,0], ref[0]) for i = n-1
        
        # In each slab, we need the 2D hypervolume of the points that are "active" in that slab.
        # A point j is active in slab i if Fv[j,0] <= x < Fv[i+1,0] for x in the slab.
        # Since Fv is sorted by obj0, all points j <= i are active in slab i.
        
        # So for slab i, the 2D points are Fv[0:i+1, 1:3]
        # We need the 2D hypervolume of these points relative to (ref[1], ref[2])
        
        # 2D HV of a set of points: sort by obj1, then for each point, the contribution is
        # (next_obj1 - obj1) * (ref[2] - min_obj2_so_far) ... no, that's not right either.
        
        # 2D HV: sort points by obj1 ascending. The hypervolume is the sum over consecutive
        # pairs of (obj1[i+1] - obj1[i]) * (ref[2] - min(obj2[0..i])) ... no.
        
        # Actually for 2D: sort by obj1. The HV is the area under the "lower envelope" in obj2.
        # Sort by obj1 ascending. For each i, the contribution is (obj1[i+1] - obj1[i]) * (ref[2] - obj2_envelope[i])
        # where obj2_envelope[i] = min(obj2[0], ..., obj2[i])
        
        # Wait, that's not right. Let me think again.
        
        # 2D HV with reference (r1, r2): for points sorted by obj1 ascending,
        # the HV is sum over i of (next_obj1 - obj1[i]) * (r2 - min_obj2_up_to_i)
        # where next_obj1 is obj1[i+1] or r2 if i is last, and min_obj2_up_to_i is min(obj2[0..i])
        
        # Hmm, that's still not quite right. Let me just compute it directly.
        
        # For 2D: the hypervolume is the area of the union of rectangles [p_i, ref) for all i.
        # Sort by obj1. The area can be computed as:
        # For i from 0 to n-1:
        #   width = (obj1[i+1] if i < n-1 else ref[1]) - obj1[i]
        #   height = ref[2] - min(obj2[j] for j <= i)  -- NO, this is wrong
        
        # Correct 2D HV: sort by obj1 ascending. 
        # hv = 0
        # min_obj2 = ref[2]
        # for i in range(n-1, -1, -1):  # process from worst to best obj1
        #   ... 
        
        # I'm overcomplicating this. Let me just use a simple but correct method for small n.
        
        # For n <= 30, I'll use the following: compute the union of boxes by
        # iterating and using a simple volume computation.
        
        # Actually, the simplest correct method for 3D HV with small n:
        # Use the fact that HV = sum over each point of its "unique" contribution.
        # A point i's unique contribution is the volume of [Fv[i], ref) that is NOT
        # covered by any other box [Fv[j], ref).
        
        # For 3D, point i's box is [f1_i, r1) x [f2_i, r2) x [f3_i, r3).
        # Point j covers part of i's box if f1_j <= f1_i, f2_j <= f2_i, f3_j <= f3_i (j dominates i)
        # or if the boxes overlap in a more complex way.
        
        # This is getting too complex. Let me just use a well-known algorithm.
        
        # I'll use the following approach for 3D HV:
        # 1. Sort by obj0
        # 2. For each slab between consecutive obj0 values, compute the 2D HV of the active points
        # 3. Sum up the contributions
        
        # For the 2D HV in each slab, I'll use a simple O(m^2) method where m is the number of active points.
        
        def _hv2(points_2d, ref2):
            """Compute 2D hypervolume. points_2d is (m, 2), ref2 is (2,)."""
            m = len(points_2d)
            if m == 0:
                return 0.0
            if m == 1:
                if points_2d[0, 0] < ref2[0] and points_2d[0, 1] < ref2[1]:
                    return (ref2[0] - points_2d[0, 0]) * (ref2[1] - points_2d[0, 1])
                return 0.0
            
            # Filter valid points
            valid = (points_2d[:, 0] < ref2[0]) & (points_2d[:, 1] < ref2[1])
            if not np.any(valid):
                return 0.0
            pts = points_2d[valid]
            
            # Sort by first coordinate
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            
            hv = 0.0
            # Process from right to left (worst to best in first coord)
            # Actually, sort by first coord ascending, then:
            # For each i, the width is pts[i+1,0] - pts[i,0] (or ref2[0] - pts[i,0] for last)
            # The height is ref2[1] - min(pts[j,1] for j >= i) ... no
            
            # Correct: sort by first coord ascending. The HV is:
            # sum over i of (x_next - x_i) * (y_ref - y_min_suffix)
            # where x_next is x[i+1] or y_ref, and y_min_suffix is min(y[i], y[i+1], ..., y[n-1])
            
            n_pts = len(pts)
            # Compute suffix minimum of y
            y_min_suffix = np.full(n_pts, ref2[1])
            for i in range(n_pts - 1, -1, -1):
                y_min_suffix[i] = min(pts[i, 1], y_min_suffix[i + 1]) if i < n_pts - 1 else pts[i, 1]
            
            for i in range(n_pts):
                x_next = pts[i + 1, 0] if i < n_pts - 1 else ref2[0]
                width = x_next - pts[i, 0]
                height = ref2[1] - y_min_suffix[i]
                if width > 0 and height > 0:
                    hv += width * height
            
            return hv
        
        # Now compute 3D HV
        n_pts = len(Fv)
        hv_total = 0.0
        
        for i in range(n_pts):
            x_next = Fv[i + 1, 0] if i < n_pts - 1 else ref[0]
            width = x_next - Fv[i, 0]
            if width <= 0:
                continue
            
            # Active points in this slab: all points j with Fv[j,0] <= Fv[i,0]
            # Since sorted, that's points 0 to i
            active = Fv[0:i+1, 1:3]
            hv2 = _hv2(active, ref[1:3])
            hv_total += width * hv2
        
        return hv_total

    def _hv_contribution_3d(self, F, ref, candidate):
        """Compute the marginal HV contribution of adding candidate to F."""
        if np.any(candidate >= ref):
            return 0.0
        
        # Check if candidate is dominated by any point in F
        if len(F) > 0:
            if np.any(np.all(F <= candidate, axis=1) & np.any(F < candidate, axis=1)):
                return 0.0
        
        F_new = np.vstack([F, candidate.reshape(1, -1)]) if len(F) > 0 else candidate.reshape(1, -1)
        hv_new = self._hv3(F_new, ref)
        hv_old = self._hv3(F, ref) if len(F) > 0 else 0.0
        return max(0.0, hv_new - hv_old)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Initialize with a diverse set of points
        init_size = min(20, budget // 10)
        archive_X = []
        archive_F = []
        evaluations = 0
        
        # Initial random points
        for _ in range(init_size):
            x = lo + (hi - lo) * np.random.random(dim)
            f = np.array(problem(x), dtype=np.float64)
            evaluations += 1
            archive_X.append(x)
            archive_F.append(f)
        
        # Remove dominated points from initial archive
        self._prune_archive(archive_X, archive_F)
        
        # Main loop: sequentially select next point to evaluate
        while evaluations < budget:
            if time.time() > deadline - 5:
                break
            
            # Strategy: for each dimension, try to find a good perturbation direction
            # by using the archive to guide the search
            
            # Select a reference point from the archive
            if len(archive_X) > 0:
                # Pick a random archive point as base
                idx = np.random.randint(len(archive_X))
                base_x = archive_X[idx].copy()
            else:
                base_x = lo + (hi - lo) * np.random.random(dim)
            
            # Generate candidate by perturbing base
            # Use a random direction with adaptive step size
            n_candidates = 5
            best_candidate = None
            best_hv_gain = -1
            
            for _ in range(n_candidates):
                # Random perturbation
                step_size = 0.1 + 0.3 * np.random.random()
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                candidate = base_x + step_size * (hi - lo) * direction
                candidate = np.clip(candidate, lo, hi)
                
                f = np.array(problem(candidate), dtype=np.float64)
                evaluations += 1
                
                if evaluations >= budget:
                    break
                
                # Compute HV gain
                hv_gain = self._hv_contribution_3d(np.array(archive_F), ref_point, f)
                
                if hv_gain > best_hv_gain:
                    best_hv_gain = hv_gain
                    best_candidate = (candidate, f)
            
            if best_candidate is not None:
                archive_X.append(best_candidate[0])
                archive_F.append(best_candidate[1])
                self._prune_archive(archive_X, archive_F)
            
            # Also add some purely random points to maintain diversity
            if evaluations < budget and evaluations % 10 == 0:
                x_rand = lo + (hi - lo) * np.random.random(dim)
                f_rand = np.array(problem(x_rand), dtype=np.float64)
                evaluations += 1
                archive_X.append(x_rand)
                archive_F.append(f_rand)
                self._prune_archive(archive_X, archive_F)
        
        return np.array(archive_X) if archive_X else None

    def _prune_archive(self, archive_X, archive_F):
        """Remove dominated points from archive in-place."""
        if len(archive_X) <= 1:
            return
        
        n = len(archive_X)
        keep = [True] * n
        
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                # Check if j dominates i
                if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                    keep[i] = False
                    break
        
        new_X = [archive_X[i] for i in range(n) if keep[i]]
        new_F = [archive_F[i] for i in range(n) if keep[i]]
        
        # Replace in place
        archive_X.clear()
        archive_F.clear()
        archive_X.extend(new_X)
        archive_F.extend(new_F)
# EVOLVE-BLOCK-END
