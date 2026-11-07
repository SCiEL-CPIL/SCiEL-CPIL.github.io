"""Rewritten program: Sobol initial sampling + hypervolume-contribution local search."""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables).
    Strategy:
    1. First 150 evaluations: deterministic Sobol quasi-random sequence for uniform coverage.
    2. Remaining 50 evaluations: hypervolume-contribution-based local search.
       - Compute marginal hypervolume contribution of each non-dominated point.
       - Replace the point with the lowest contribution via Gaussian perturbations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        eval_count = 0
        max_evals = self.budget

        def eval_problem(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return np.array(problem(x))

        X_list = []
        F_list = []

        # Phase 1: Sobol quasi-random initial sampling (150 points)
        n_sobol = min(150, max_evals)
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            samples = sampler.random(n_sobol)
        except Exception:
            samples = np.random.rand(n_sobol, self.dim)

        X_init = lo + samples * span

        for x in X_init:
            if eval_count >= max_evals:
                break
            f = eval_problem(x)
            X_list.append(x.copy())
            F_list.append(f.copy())

        # Helper: check if f is dominated by any point in F_arr
        def is_dominated(f, F_arr):
            if len(F_arr) == 0:
                return False
            diffs = F_arr - f
            return np.any(np.all(diffs <= 0, axis=1) & np.any(diffs < 0, axis=1))

        # Helper: get non-dominated indices
        def get_non_dominated_indices(F_arr):
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Helper: compute hypervolume contribution of a single point given the set of non-dominated points
        # Reference point: [1705.0, 12.0, 0.3]
        ref = np.array([1705.0, 12.0, 0.3])

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume in 3D for a set of non-dominated points."""
            if len(points) == 0:
                return 0.0
            # Filter points that are strictly better than reference in all objectives
            valid = np.all(points < ref, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(points[:, 0])
            points = points[order]
            
            # For 3D, we can use a sweep-line approach
            # Hypervolume = sum over each point of the volume of the box it uniquely contributes
            # A point p contributes the box [p[0], next_p[0]) x [p[1], next_p[1]) x [p[2], ref[2])
            # but only if no other point dominates that box region
            
            # Simpler approach for small n: enumerate boxes
            n = len(points)
            hv = 0.0
            for i in range(n):
                p = points[i]
                # The box for point i extends from p to ref, but is clipped by other points
                # that are better in some objectives
                x_max = ref[0]
                y_max = ref[1]
                z_max = ref[2]
                
                # Find the maximum extent in each dimension considering other points
                # Point i's unique contribution is the region where no other point j
                # has j[k] <= p[k] for all k where j differs
                # This is complex; use the standard algorithm:
                # Sort by x, then for each point, the contribution is:
                # (x_next - x_i) * (area of 2D hypervolume of points with x > x_i and y,z < ref)
                
                # Let's use a simpler O(n^2) approach for small n:
                # For each point, compute the volume of the box [p, ref) that is not covered by any other point
                # This is equivalent to: for point i, the box is [p_i, ref) minus the union of boxes [p_j, ref) for j != i
                # But since all points are non-dominated, the boxes only overlap in specific ways
                
                # Actually, for non-dominated points in 3D, we can use:
                # Sort by x ascending. For point i, its contribution is:
                # dx = (x_{i+1} - x_i) if i < n-1 else (ref[0] - x_i)
                # dy = min of (ref[1] - y_i) and (y_j - y_i) for all j with x_j > x_i and y_j < y_i... 
                # This is getting complex. Let me use a different approach.
                
                # Use the "corner" method: each non-dominated point contributes a rectangular box
                # where the box extends from the point to the reference, but is truncated by
                # the next better point in each dimension.
                
                # For simplicity and correctness with small n, use inclusion-exclusion or
                # just compute the volume of the box [p_i, ref) and subtract overlaps.
                # But overlaps are tricky. Let's use a sweep approach.
                
                pass
            
            # Let me implement a proper 3D hypervolume calculation
            # Using the algorithm: sort by x, then use a 2D hypervolume for the remaining dimensions
            # with appropriate clipping
            
            # Sort by x
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            
            total_hv = 0.0
            for i in range(len(pts)):
                p = pts[i]
                # x range: from p[0] to next point's x or ref[0]
                x_end = pts[i+1][0] if i+1 < len(pts) else ref[0]
                dx = x_end - p[0]
                if dx <= 0:
                    continue
                
                # For the 2D projection (y, z), we need the hypervolume of the region
                # where y < ref[1], z < ref[2], and no point with x >= p[0] dominates
                # Actually, points with x >= p[0] that are better in y and z clip this region
                # But since we sorted by x, all remaining points have x >= p[0]
                
                # The 2D region is clipped by points j with x_j >= p[0] (which is all remaining points)
                # that have y_j < ref[1] and z_j < ref[2]
                # The 2D hypervolume of the non-dominated set in (y,z) projected from points i..end
                
                # For points from i to end, consider their (y,z) coordinates
                sub_pts = pts[i:, 1:3]
                ref_2d = ref[1:3]
                
                # Filter valid points
                valid_2d = np.all(sub_pts < ref_2d, axis=1)
                sub_pts = sub_pts[valid_2d]
                
                if len(sub_pts) == 0:
                    continue
                
                # 2D hypervolume: sort by y, then sum (y_next - y_i) * (ref_z - z_i) for non-dominated in 2D
                # But we need the hypervolume of the union of boxes [p_j, ref_2d) for j in sub_pts
                # For non-dominated points in 2D, this is straightforward
                
                # Get non-dominated points in 2D
                nd_2d = get_non_dominated_indices(sub_pts)
                sub_pts_nd = sub_pts[nd_2d]
                
                if len(sub_pts_nd) == 0:
                    continue
                
                # Sort by y
                order_y = np.argsort(sub_pts_nd[:, 0])
                sub_pts_nd = sub_pts_nd[order_y]
                
                hv_2d = 0.0
                for k in range(len(sub_pts_nd)):
                    py = sub_pts_nd[k][0]
                    pz = sub_pts_nd[k][1]
                    y_end = sub_pts_nd[k+1][0] if k+1 < len(sub_pts_nd) else ref_2d[0]
                    dy = y_end - py
                    if dy <= 0:
                        continue
                    dz = ref_2d[1] - pz
                    if dz <= 0:
                        continue
                    hv_2d += dy * dz
                
                total_hv += dx * hv_2d
            
            return total_hv

        def hv_contribution(point, F_nd, ref):
            """Compute the marginal hypervolume contribution of a point."""
            # Hypervolume with the point minus hypervolume without
            # F_nd is the set of non-dominated points including this point
            hv_with = hypervolume_3d(F_nd, ref)
            # Remove this point
            mask = np.ones(len(F_nd), dtype=bool)
            # Find the index of this point
            for i in range(len(F_nd)):
                if np.allclose(F_nd[i], point, atol=1e-10):
                    mask[i] = False
                    break
            F_nd_without = F_nd[mask]
            # Recompute non-dominated without this point (it might make others dominated)
            if len(F_nd_without) > 0:
                nd_idx = get_non_dominated_indices(F_nd_without)
                F_nd_without = F_nd_without[nd_idx]
            hv_without = hypervolume_3d(F_nd_without, ref)
            return hv_with - hv_without

        # Phase 2: Hypervolume-contribution-based local search
        # Use remaining budget for refinement
        n_refine = max_evals - eval_count
        if n_refine > 0 and time.time() < deadline:
            # Get current non-dominated set
            F_arr = np.array(F_list)
            nd_indices = get_non_dominated_indices(F_arr)
            
            if len(nd_indices) > 0:
                # Compute contributions for each non-dominated point
                contributions = []
                for idx in nd_indices:
                    contrib = hv_contribution(F_arr[idx], F_arr[nd_indices], ref)
                    contributions.append(contrib)
                contributions = np.array(contributions)
                
                # Perform local search for remaining evaluations
                for step in range(n_refine):
                    if eval_count >= max_evals:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Recompute non-dominated set and contributions
                    F_arr = np.array(F_list)
                    nd_indices = get_non_dominated_indices(F_arr)
                    if len(nd_indices) == 0:
                        break
                    
                    # Find the point with lowest contribution
                    contributions = []
                    for idx in nd_indices:
                        contrib = hv_contribution(F_arr[idx], F_arr[nd_indices], ref)
                        contributions.append(contrib)
                    contributions = np.array(contributions)
                    
                    # Pick the point with lowest contribution
                    worst_local = np.argmin(contributions)
                    worst_idx = nd_indices[worst_local]
                    x_worst = X_list[worst_idx]
                    
                    # Generate Gaussian perturbation
                    # Scale by the span of the variable space
                    sigma = 0.1 * span  # 10% of the range
                    x_new = x_worst + np.random.normal(0, sigma)
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = eval_problem(x_new)
                    X_list.append(x_new.copy())
                    F_list.append(f_new.copy())
                    
                    # If the new point is non-dominated, it stays; otherwise it's just recorded
                    # We don't remove dominated points from the list since all evaluations are recorded

        return X_list
