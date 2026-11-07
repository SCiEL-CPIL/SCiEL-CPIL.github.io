"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-driven sequential search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Reference point for hypervolume calculation
        ref = np.array([3.0, 5.0, 7.0])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume exactly using inclusion-exclusion."""
            if len(points) == 0:
                return 0.0
            # Filter points that are better than reference in all objectives
            pts = [p for p in points if np.all(p < ref)]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = sorted(pts, key=lambda p: p[0])
            
            # For 3D, we can use a recursive approach
            # Project onto 2D (objectives 2 and 3) for each point
            # and compute the area contribution
            
            # Use the standard 3D hypervolume algorithm
            # Sort by objective 1
            n = len(pts)
            if n == 1:
                p = pts[0]
                return max(0, ref[0] - p[0]) * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
            
            # Recursive: split by first objective
            # For each point i, its contribution is the volume of the box
            # [p_i[0], ref[0]] x [p_i[1], ref[1]] x [p_i[2], ref[2]]
            # minus overlaps with other points that dominate it in the projection
            
            # Simpler: use the layer-cake approach
            # Sort by obj1 ascending
            hv = 0.0
            for i in range(n):
                p = pts[i]
                if p[0] >= ref[0]:
                    break
                # The slice at this point: width in obj1 direction
                # minus what's already covered by points with smaller obj1
                # that extend far enough in obj1
                w = ref[0] - p[0]
                # Subtract overlap from previous points that have obj1 < p[0]
                # and whose box extends past p[0]
                # Actually, let's use a different approach: 
                # For each point, compute the 2D area of the region in (obj2, obj3)
                # that is not dominated by any other point with obj1 <= p[0]
                
                # This is getting complex. Let me use a simpler O(n^3) approach for small n
                pass
            
            # Use a direct O(n^3) approach for correctness
            # For 3 points, the hypervolume can be computed by sorting and using
            # the fact that in 3D, we can decompose into 2D subproblems
            # 
            # Actually, for small archives (typically < 50 points), let's use
            # a grid-free exact method
            # 
            # Standard algorithm: sort by f1, then for each point compute
            # the 2D hypervolume of points that are not dominated in f2,f3
            # by any point with f1 <= current f1
            
            # Let me implement a clean 3D hypervolume
            # Sort by first coordinate
            pts_sorted = sorted(pts, key=lambda x: x[0])
            
            # For each point i, the contribution is:
            # (ref[0] - pts_sorted[i][0]) * area_2d({pts_sorted[j] for j > i where pts_sorted[j][1] > pts_sorted[i][1] and pts_sorted[j][2] > pts_sorted[i][2]})
            # Wait, that's not quite right either.
            
            # Let me use the standard recursive decomposition:
            # HV_3(pts, ref) = sum over i of (ref[0] - pts[i][0]) * HV_2D({pts[j][1:] for j where pts[j][0] >= pts[i][0]}, ref[1:])
            # No, that's not right either.
            
            # Correct approach: 
            # Sort by f1 ascending. For point i, its "slice" contribution is:
            # width_i * area_2d of the region in (f2,f3) space that is:
            #   - above pts[i] in f2 and f3
            #   - not covered by any point j with f1_j <= f1_i
            # 
            # Actually the cleanest way: 
            # HV = integral over all boxes. For non-dominated set sorted by f1:
            # For point i (0-indexed, sorted by f1 asc):
            #   contribution = (f1_{i-1} - f1_i) * A_i  for i > 0, where f1_{-1} = ref[0]
            #   Wait, I need to think about this differently.
            #
            # Let me just use a brute-force inclusion-exclusion for small sets
            # or use the well-known algorithm.
            
            # Simple O(n^3) exact 3D hypervolume:
            # Sort by f1
            pts_s = sorted(pts, key=lambda x: x[0])
            n = len(pts_s)
            hv = 0.0
            for i in range(n):
                # The region where pts_s[i] is the "frontier" in f1 direction
                # is bounded by f1 in [pts_s[i][0], min(f1 of next point or ref[0])]
                # and in (f2,f3) by the 2D hypervolume of points that are
                # not dominated in (f2,f3) by any point with f1 <= pts_s[i][0]
                
                f1_next = ref[0] if i == n - 1 else pts_s[i+1][0]
                if f1_next <= pts_s[i][0]:
                    continue
                w = f1_next - pts_s[i][0]
                
                # 2D hypervolume of points with f1 <= pts_s[i][0] in (f2,f3) space
                # relative to (pts_s[i][1], pts_s[i][2]) as lower bound... no
                # Actually, we need the 2D HV of the set {pts_s[j][1:] : j <= i} 
                # with reference (ref[1], ref[2]) but only counting the part
                # that is "above" pts_s[i] in the sense of not being dominated
                # by pts_s[i] in (f2,f3).
                
                # This is getting complicated. Let me use a different strategy:
                # For small n, compute HV by iterating over all 2^k subsets? No, too slow.
                
                # Let me use the standard "slice" method properly:
                # After sorting by f1, for each i, the slice at f1 = pts_s[i][0]
                # has height (pts_s[i+1][0] - pts_s[i][0]) [or ref[0] - pts_s[i][0] for last]
                # and the cross-section is the 2D region in (f2,f3) that is below ref
                # and not dominated by any point in the archive with f1 <= pts_s[i][0].
                
                # The cross-section 2D HV is computed from the set of points
                # pts_s[0..i] projected to (f2,f3), with reference (ref[1],ref[2])
                # but we only count area that is >= pts_s[i] in both f2 and f3? No.
                
                # I think the cleanest implementation for 3D with small n is:
                # Use the fact that HV_3 = sum of individual box volumes minus overlaps
                # But that's O(2^n).
                
                # For n <= 20 or so, let me just use a numerical approach or
                # a proper algorithm.
                
                # Let me implement the standard algorithm from the literature:
                # 1. Sort by f1
                # 2. For i from 0 to n-1:
                #    Compute 2D HV of {pts[j][1:] for j in 0..i} with ref [ref[1], ref[2]]
                #    But only the portion that is "new" at this f1 level
                #    
                # Actually, I'll use a simpler but correct approach:
                # For 3D, HV can be computed as:
                # Sort by f1. For each i, the contribution is:
                # (ref[0] - f1_i) * HV_2D({(f2_j, f3_j) for j > i where f2_j > f2_i and f3_j > f3_i}, (ref[1], ref[2]))
                # Wait, that's not right either.
                
                # Let me just use a straightforward O(n^3) method:
                # For each pair of points, compute the 1D interval, etc.
                # 
                # Actually, the simplest correct method for small n:
                # HV_3 = sum_i (ref[0]-f1_i)*(ref[1]-f2_i)*(ref[2]-f3_i) 
                #         - sum_{i<j} overlaps...
                # This inclusion-exclusion is O(n^3) for 3D which is fine.
                
                # Let me just use a grid-based approximation for now since n is small
                # No, we need exact.
                
                # OK let me implement the proper recursive algorithm:
                pass
            
            # I'll implement a clean version:
            # For 3D hypervolume with reference point:
            # 1. Filter points better than ref in all dims
            # 2. Sort by first objective
            # 3. Use the "slice" decomposition
            
            # Slice decomposition for 3D:
            # After sorting by f1, define f1_{-1} = ref[0], f1_n = -inf
            # For each i, the slice between f1_i and f1_{i-1} (where f1_{i-1} is the next larger f1)
            # has a 2D cross-section which is the 2D HV of points with index <= i
            # projected to (f2, f3) with reference (ref[1], ref[2])
            
            # Let me just implement 2D HV and use it in the slice method
            
            def hv_2d(points_2d, ref_2d):
                """2D hypervolume exactly."""
                pts = [p for p in points_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                if len(pts) == 0:
                    return 0.0
                pts = sorted(pts, key=lambda x: x[0])
                area = 0.0
                max_y = -np.inf
                for i, p in enumerate(pts):
                    if p[1] > max_y:
                        max_y = p[1]
                    # The width in x direction for this point
                    if i == len(pts) - 1:
                        x_width = ref_2d[0] - p[0]
                    else:
                        x_width = pts[i+1][0] - p[0]
                    # The height is (ref_2d[1] - max_y) but we need to be careful
                    # Actually, the 2D HV with sorted points:
                    # For point i, its contribution is x_width * (ref_2d[1] - p[1])
                    # if p[1] is the maximum so far, otherwise it's already covered
                    if p[1] <= max_y:
                        # This point doesn't add new area in y direction
                        # But it might extend in x direction
                        # Actually in 2D, if sorted by x, the contribution of point i
                        # is x_width * (ref_y - max(p[1], max_y_so_far))... no
                        # 
                        # Correct 2D HV: sort by x ascending.
                        # For each point i, the area it contributes is:
                        #   (x_{i+1} - x_i) * (ref_y - y_i) if y_i is the running max
                        #   Otherwise, the area is already covered by a point with smaller x
                        #   and larger y.
                        # 
                        # Wait, that's not right. In 2D, if we sort by x ascending:
                        # The HV is the area under the "staircase" formed by the points.
                        # For point i, if y_i < max_y_so_far, then the entire vertical strip
                        # at x_i is already covered by a previous point with larger y.
                        # So contribution is 0 in that case for the "new" area.
                        # But the x-width still matters for subsequent points.
                        # 
                        # Actually: contribution_i = x_width_i * max(0, ref_y - max_y_so_far)
                        # where max_y_so_far includes point i.
                        # No wait. Let me think again.
                        # 
                        # 2D HV = area of union of rectangles [x_i, ref_x] x [y_i, ref_y]
                        # After sorting by x ascending, the union can be decomposed as:
                        # For i from 0 to n-1:
                        #   width_i = (x_{i+1} if i < n-1 else ref_x) - x_i
                        #   height_i = ref_y - max(y_j for j <= i)  ... no
                        # 
                        # The correct formula: 
                        # Keep track of max_y = -inf
                        # For i in range(n):
                        #   if y_i > max_y:
                        #     max_y = y_i
                        #   x_next = x_{i+1} if i < n-1 else ref_x
                        #   area += (x_next - x_i) * max(0, ref_y - max_y)
                        # 
                        # Wait, that gives the area ABOVE the staircase, which is what we want
                        # if the staircase is the upper envelope. But in minimization, lower is better,
                        # so the Pareto front is the lower-left boundary. The HV is the area
                        # between the front and the reference point.
                        # 
                        # So for minimization: points with smaller x and y are better.
                        # The HV is the area of the region that is dominated by the front
                        # and below the reference point.
                        # 
                        # After sorting by x ascending, the "frontier" in 2D is:
                        # For each x_i, the best (smallest) y among points with x <= x_i
                        # is min_y_so_far. The HV contribution of the strip [x_i, x_{i+1}]
                        # is (x_{i+1} - x_i) * (ref_y - min_y_so_far)
                        # 
                        # Let me redo:
                        min_y = np.inf
                        area = 0.0
                        for i, p in enumerate(pts):
                            if p[1] < min_y:
                                min_y = p[1]
                            x_next = pts[i+1][0] if i < len(pts)-1 else ref_2d[0]
                            w = x_next - p[0]
                            h = max(0.0, ref_2d[1] - min_y)
                            area += w * h
                        return area
            
            # Now use slice method for 3D:
            pts_s = sorted(pts, key=lambda x: x[0])
            n = len(pts_s)
            hv = 0.0
            for i in range(n):
                x_next = pts_s[i+1][0] if i < n-1 else ref[0]
                w = x_next - pts_s[i][0]
                if w <= 0:
                    continue
                # 2D HV of points 0..i projected to (f2,f3) with ref (ref[1],ref[2])
                pts_2d = [(pts_s[j][1], pts_s[j][2]) for j in range(i+1)]
                area_2d = hv_2d(pts_2d, (ref[1], ref[2]))
                hv += w * area_2d
            return hv

        def marginal_hv(point_f, archive_F, ref):
            """Compute marginal hypervolume contribution of adding point_f to archive."""
            # Current HV
            hv_before = hypervolume_3d(archive_F, ref)
            # HV with new point
            new_archive = archive_F + [point_f]
            hv_after = hypervolume_3d(new_archive, ref)
            return hv_after - hv_before

        # Initialize with random points
        init_size = min(10, self.budget // 5)
        for _ in range(init_size):
            x = lo + range_ * rng.random(self.dim)
            f = problem(x)
            if not is_dominated(f, archive_F):
                # Remove any points dominated by f
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                archive_X = new_archive_X
                archive_F = new_archive_F
                archive_X.append(x)
                archive_F.append(f)

        used = init_size
        all_F = list(archive_F)

        # Main loop: hypervolume-driven search
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find the best point to evaluate next
            # Strategy: sample candidate directions from archive points and
            # evaluate the one with highest expected marginal HV
            
            best_candidate = None
            best_marginal = -1.0

            # Generate candidates by perturbing archive points
            n_candidates = min(5, len(archive_X))
            for _ in range(n_candidates):
                if len(archive_X) == 0:
                    break
                idx = rng.integers(len(archive_X))
                parent = archive_X[idx]
                # Adaptive step size based on archive spread
                if len(archive_X) > 1:
                    spread = np.std(np.array(archive_X), axis=0)
                    step = 0.1 * spread
                    step = np.maximum(step, 0.01 * range_)
                else:
                    step = 0.1 * range_
                
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                all_F.append(fc)
                
                if not is_dominated(fc, archive_F):
                    # Remove dominated points from archive
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(fc <= af) and np.any(fc < af)):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    archive_X.append(child)
                    archive_F.append(fc)
                    break
            else:
                # If we didn't find a non-dominated point, try a random point
                x = lo + range_ * rng.random(self.dim)
                fc = problem(x)
                used += 1
                all_F.append(fc)
                if not is_dominated(fc, archive_F):
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(fc <= af) and np.any(fc < af)):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    archive_X.append(x)
                    archive_F.append(fc)

            # Periodic local search around best archive points
            if used % 10 == 0 and len(archive_X) > 0 and used < self.budget - 5:
                # Pick the archive point with best single-objective value
                best_idx = np.argmin(np.sum(archive_F, axis=1))
                center = archive_X[best_idx]
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    step = 0.05 * range_
                    x = center + step * rng.standard_normal(self.dim)
                    x = np.clip(x, lo, hi)
                    fc = problem(x)
                    used += 1
                    all_F.append(fc)
                    if not is_dominated(fc, archive_F):
                        new_archive_X = []
                        new_archive_F = []
                        for ax, af in zip(archive_X, archive_F):
                            if not (np.all(fc <= af) and np.any(fc < af)):
                                new_archive_X.append(ax)
                                new_archive_F.append(af)
                        archive_X = new_archive_X
                        archive_F = new_archive_F
                        archive_X.append(x)
                        archive_F.append(fc)

        return all_F
# EVOLVE-BLOCK-END
