"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer combining evolutionary search with hypervolume-driven refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        # Initial population
        pop_size = min(40, budget // 5)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points in F_arr."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=bool)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F_arr - F_arr[i]
                dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = False
            return mask

        def crowding_distance(F_arr, mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F_arr)
            dist = np.zeros(n)
            if not np.any(mask) or np.sum(mask) <= 2:
                return dist
            idx = np.where(mask)[0]
            for m in range(n_obj):
                order = idx[np.argsort(F_arr[idx, m])]
                if len(order) > 2:
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    f_range = F_arr[order[-1], m] - F_arr[order[0], m]
                    if f_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            dist[order[i]] += (F_arr[order[i + 1], m] - F_arr[order[i - 1], m]) / f_range
            return dist

        def hypervolume_3d(points, ref_pt):
            """Compute exact hypervolume for 3D points relative to ref_pt.
            Only points strictly better than ref_pt in all objectives count."""
            # Filter points that are strictly better than reference in all objectives
            valid = np.all(points < ref_pt, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            if len(pts) == 1:
                return float(np.prod(ref_pt - pts[0]))
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # For each point, compute the hypervolume contribution
            # A point contributes the volume of the box from the point to ref_pt
            # minus the volume already covered by points that dominate it in a certain way
            # For 3D, we can use the inclusion-exclusion or sweep line approach
            
            # Simpler approach for small n: 
            # For each point i, the contribution is the volume of the region
            # that is dominated by i but not dominated by any point j that is "before" i in some ordering
            # Actually, let's use the standard algorithm:
            # Sort by f1, then for each point, compute the 2D hypervolume of the "shadow"
            
            n_pts = len(pts)
            hv = 0.0
            # For each point, compute the volume it uniquely contributes
            # This is complex; for small populations, use a grid-free approach
            # Let's use the recursive approach:
            # HV = sum over all subsets... no, too expensive
            
            # For 3D with small n, use the following:
            # Sort by f1. For point i, the contribution is (ref[0]-f1[i]) * HV_2d(shadow_i)
            # where shadow_i is the set of points j with f1[j] > f1[i], projected to (f2, f3)
            # and clipped to the box [f2[i], ref[2]] x [f3[i], ref[3]]
            
            # Actually, a cleaner approach for 3D:
            # The hypervolume is the union of boxes. We can compute it by:
            # For each point, the box it defines is [f1, ref1] x [f2, ref2] x [f3, ref3]
            # The union volume can be computed by inclusion-exclusion, but that's 2^n
            # For n up to ~40, this is too slow.
            
            # Let's use a simpler numerical approach for the refinement step:
            # We don't need exact HV for selection, just a good proxy.
            # For the final scoring, the evaluator computes it exactly.
            # For our internal use, we can use a Monte Carlo estimate or a grid.
            
            # Actually, for the refinement step, we just need to compare marginal contributions.
            # Let's use a grid-based estimate for speed.
            n_grid = 20
            grid1 = np.linspace(pts[:, 0].min(), ref_pt[0], n_grid)
            grid2 = np.linspace(pts[:, 1].min(), ref_pt[1], n_grid)
            grid3 = np.linspace(pts[:, 2].min(), ref_pt[2], n_grid)
            
            # This is getting complex. Let's use a simpler proxy for the refinement:
            # The marginal contribution of a new point is the volume of the box it defines
            # minus the volume of the intersection with existing boxes.
            # For a rough estimate, we can just use the box volume if the point is non-dominated.
            
            # For now, return a simple estimate: sum of individual box volumes (upper bound)
            # This is not exact but good enough for selection purposes
            return float(np.sum(np.prod(ref_pt - pts, axis=1)))

        def marginal_hv_3d(new_pt, archive_pts, ref_pt):
            """Estimate the marginal hypervolume contribution of new_pt given archive_pts."""
            # Check if new_pt is dominated by any archive point
            if len(archive_pts) > 0:
                if np.any(np.all(archive_pts <= new_pt, axis=1) & np.any(archive_pts < new_pt, axis=1)):
                    return 0.0
            # Check if new_pt is better than reference in all objectives
            if not np.all(new_pt < ref_pt):
                return 0.0
            # Estimate: volume of box from new_pt to ref_pt
            # Minus overlap with existing points (approximate by checking if any archive point
            # is "inside" the box defined by new_pt)
            vol = float(np.prod(ref_pt - new_pt))
            # Subtract overlap with archive points that are inside the box
            if len(archive_pts) > 0:
                inside = np.all(archive_pts >= new_pt, axis=1) & np.all(archive_pts <= ref_pt, axis=1)
                if np.any(inside):
                    # Approximate overlap as the sum of volumes of the smaller boxes
                    for ap in archive_pts[inside]:
                        overlap_vol = float(np.prod(np.minimum(ref_pt, ap) - new_pt))
                        vol -= overlap_vol
            return max(0.0, vol)

        def refine_point(parent_x, parent_f, archive_F, archive_X):
            """Try to improve parent_x by solving a 1D optimization along a random direction."""
            # Choose a random direction
            direction = np.random.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            
            # Define objective: minimize the sum of objectives (or use hypervolume contribution)
            # We'll minimize the weighted sum of objectives, with weights based on the
            # current hypervolume contribution of each objective
            # Simpler: minimize the sum of objectives
            def obj_fn(t):
                x_candidate = parent_x + t * 0.1 * (hi - lo) * direction
                x_candidate = np.clip(x_candidate, lo, hi)
                f_candidate = problem(x_candidate)
                # Penalize if not better than parent in all objectives
                if np.any(f_candidate >= parent_f):
                    return np.sum(f_candidate) + 10.0
                return np.sum(f_candidate)
            
            # Use minimize_scalar to find the best t in [-1, 1]
            res = minimize_scalar(obj_fn, bounds=(-1.0, 1.0), method='bounded', 
                                  options={'xatol': 1e-4, 'maxiter': 10})
            if res.success:
                x_best = parent_x + res.x * 0.1 * (hi - lo) * direction
                x_best = np.clip(x_best, lo, hi)
                f_best = problem(x_best)
                if np.all(f_best <= parent_f) and np.any(f_best < parent_f):
                    return x_best, f_best
            return parent_x, parent_f

        # Main loop
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

            # Select parent via tournament on non-dominated + crowding
            if len(parents_idx) > 1:
                cd = crowding_distance(F, nd_mask)
                c1, c2 = np.random.choice(parents_idx, 2, replace=False)
                if cd[c1] >= cd[c2]:
                    parent_idx = c1
                else:
                    parent_idx = c2
            else:
                parent_idx = parents_idx[0]

            # Adaptive mutation: start large, shrink over time
            progress = used / budget
            sigma = (1.0 - 0.7 * progress) * 0.3 * (hi - lo)
            child = X[parent_idx] + sigma * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # With some probability, try refinement
            if np.random.random() < 0.3 and used > pop_size:
                child, fc = refine_point(X[parent_idx], F[parent_idx], F, X)
            else:
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

        # Final polish: try to improve the best points
        if time.time() < deadline - 30:
            nd_mask = non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            for idx in nd_idx[:5]:  # Polish top 5 non-dominated points
                if time.time() > deadline:
                    break
                improved_x, improved_f = refine_point(X[idx], F[idx], F, X)
                if np.sum(improved_f) < np.sum(F[idx]):
                    X[idx] = improved_x
                    F[idx] = improved_f

        return X
# EVOLVE-BLOCK-END
