"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sequential hypervolume-guided local search for WFG7 (3-objective)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([3.0, 5.0, 7.0])

        # --- Phase 1: Latin Hypercube Sampling for initial diverse set ---
        # Use a small initial population to cover the space
        n_init = min(20, budget // 10)
        X = []
        F = []
        used = 0

        # LHS: divide each dimension into n_init equal intervals
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                # Random point within the i-th interval of dimension d
                interval_start = i / n_init
                interval_end = (i + 1) / n_init
                u = rng.random()
                x[d] = lo[d] + (hi[d] - lo[d]) * (interval_start + u * (interval_end - interval_start))
            X.append(x)
            f = np.array(problem(x), dtype=float)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # --- Helper: compute non-dominated set ---
        def get_non_dominated_indices(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # --- Helper: compute hypervolume contribution of a single point given other points ---
        # For 3 objectives, we can compute the contribution efficiently
        def hv_contribution(point, others_F, ref):
            """Compute the hypervolume that point contributes given the set others_F (excluding point)."""
            # A point contributes if it is not dominated by any point in others_F
            # and its contribution is the volume of the region it uniquely covers
            
            # Check if point is dominated by any in others_F
            if len(others_F) > 0:
                # If any other point dominates this point, contribution is 0
                dominates = np.all(others_F <= point, axis=1) & np.any(others_F < point, axis=1)
                if np.any(dominates):
                    return 0.0
            
            # Compute the hypervolume of the region where this point is the best in at least one objective
            # For 3D, the contribution is the volume of the box [point, ref] minus the volume
            # covered by other points that are "below" this point in the dominance sense
            
            # Simple approach: compute the full box volume, then subtract overlaps
            # Box volume for this point
            box = np.maximum(ref - point, 0)
            if np.any(box <= 0):
                return 0.0
            full_vol = box[0] * box[1] * box[2]
            
            # Subtract volumes covered by other points that are "dominated by" or "overlap with" this point
            # A point q overlaps with p if q is in the box [p, ref] or p is in [q, ref]
            # For contribution, we need to find the region where p is the minimum in at least one dimension
            
            # More precise: the contribution of p is the volume of {x : p <= x <= ref} 
            # minus the union of {q <= x <= ref} for all q that are not dominated by p
            
            # This is complex. Let's use a simpler incremental approach:
            # Compute HV of {point} + others_F minus HV of others_F
            
            def hv_3d(points, ref):
                """Compute exact 3D hypervolume."""
                if len(points) == 0:
                    return 0.0
                # Filter points that are within the reference box
                pts = []
                for p in points:
                    if np.all(p < ref):
                        pts.append(p)
                if len(pts) == 0:
                    return 0.0
                pts = np.array(pts)
                # Non-dominated filtering
                n = len(pts)
                nd = np.ones(n, dtype=bool)
                for i in range(n):
                    if not nd[i]:
                        continue
                    for j in range(n):
                        if i == j:
                            continue
                        if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                            nd[i] = False
                            break
                pts = pts[nd]
                if len(pts) == 0:
                    return 0.0
                
                # Sort by first objective
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                
                # Use the inclusion-exclusion or sweep-line approach for 3D
                # For small sets, we can use a simple approach
                # Compute the union of boxes [p, ref]
                
                # Use a simple recursive approach for small n
                if len(pts) <= 1:
                    box = ref - pts[0]
                    return max(0, box[0]) * max(0, box[1]) * max(0, box[2])
                
                # For larger sets, use a divide-and-conquer or grid-based approach
                # Here we use a simple but correct method: compute the volume by
                # sweeping through sorted points
                
                # Actually, let's use a more robust method for small n:
                # The hypervolume is the volume of the union of axis-aligned boxes
                # We can compute this by inclusion-exclusion for very small n,
                # or by a sweep-line approach
                
                # For n up to ~20, let's use a simple approach:
                # Sort by x, then for each point, compute the area in yz-plane
                # that is not covered by subsequent points (in x-order)
                
                hv = 0.0
                for i in range(len(pts)):
                    # Box from pts[i] to ref
                    x_min = pts[i, 0]
                    y_min = pts[i, 1]
                    z_min = pts[i, 2]
                    x_max = ref[0]
                    y_max = ref[1]
                    z_max = ref[2]
                    
                    # Width in x for this segment
                    x_width = x_max - x_min
                    if x_width <= 0:
                        continue
                    
                    # In the yz-plane, we need the area where y >= y_min and z >= z_min
                    # that is not covered by any other point j > i (in x-order)
                    # with y_j <= y and z_j <= z for some region
                    
                    # For simplicity, compute the full area and subtract overlaps
                    full_area = (y_max - y_min) * (z_max - z_min)
                    if full_area <= 0:
                        continue
                    
                    # Subtract areas covered by other points
                    # This is getting complex. Let's use a different approach.
                    
                    break  # Fallback to a simpler method
                
                # Let's use a grid-based approximation for the contribution calculation
                # But for exactness, let's implement a proper 3D HV
                
                # Proper 3D HV using the "divide and conquer" approach
                return self._hv_3d_exact(pts, ref)
            
            hv_with = hv_3d(np.vstack([point.reshape(1, -1), others_F]), ref)
            hv_without = hv_3d(others_F, ref) if len(others_F) > 0 else 0.0
            return max(0, hv_with - hv_without)
        
        def _hv_3d_exact(self, pts, ref):
            """Exact 3D hypervolume for a set of non-dominated points."""
            if len(pts) == 0:
                return 0.0
            if len(pts) == 1:
                box = ref - pts[0]
                return max(0, box[0]) * max(0, box[1]) * max(0, box[2])
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Use the standard 3D HV algorithm
            # The hypervolume can be computed by sweeping in x and maintaining
            # the 2D hypervolume in yz-plane
            
            # For small n, use a simple but correct method:
            # HV = sum over i of (x_{i+1} - x_i) * HV_2D(points in [x_i, x_{i+1}])
            # where x_0 = min(pts[:,0]), x_{n+1} = ref[0]
            
            x_vals = np.concatenate([pts[:, 0], [ref[0]]])
            hv = 0.0
            
            for i in range(len(pts)):
                x_left = pts[i, 0]
                x_right = x_vals[i + 1]
                if x_right <= x_left:
                    continue
                x_width = x_right - x_left
                
                # Find all points that are "active" in this x-slice
                # A point j is active if pts[j, 0] <= x_left (it extends to the right)
                # Actually, point j contributes to the slice [x_left, x_right] if pts[j, 0] <= x_left
                # and ref[0] >= x_right (always true)
                
                # The yz-region covered by point j in this slice is [pts[j,1], ref[1]] x [pts[j,2], ref[2]]
                # We need the 2D hypervolume of these rectangles
                
                active = pts[:i + 1]  # Points with x <= x_left (since sorted)
                # But we need points that are not dominated in yz by others in the active set
                # For the 2D HV, we just need the non-dominated points in yz
                
                # Filter active points to those with y < ref[1] and z < ref[2]
                active = active[np.all(active[:, 1:] < ref[1:], axis=1)]
                if len(active) == 0:
                    continue
                
                # Compute 2D hypervolume in yz-plane
                hv_2d = self._hv_2d(active[:, 1:], ref[1:])
                hv += x_width * hv_2d
            
            return hv
        
        def _hv_2d(self, pts, ref):
            """Exact 2D hypervolume."""
            if len(pts) == 0:
                return 0.0
            # Sort by first coordinate
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Remove dominated points in 2D
            n = len(pts)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and \
                       (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                        nd[i] = False
                        break
            pts = pts[nd]
            if len(pts) == 0:
                return 0.0
            
            # Sort by x
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            y_max = ref[1]
            for i in range(len(pts)):
                x_left = pts[i, 0]
                x_right = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
                x_width = x_right - x_left
                if x_width <= 0:
                    continue
                # The y-extent covered by point i and all previous points
                # Since sorted by x, point i has the smallest x among remaining
                # The y-extent is from min y of active points to ref[1]
                # Active points are those with x <= x_left, which is all points up to i
                # The minimum y among these is pts[i, 1] (since we're going in order of increasing x,
                # and for non-dominated points, y decreases as x increases)
                y_min = pts[i, 1]
                y_width = y_max - y_min
                if y_width <= 0:
                    continue
                hv += x_width * y_width
            return hv

        # --- Phase 2: Iterative hypervolume-guided local search ---
        # For each non-dominated point, compute its marginal HV contribution
        # Select the one with highest contribution and refine it via Nelder-Mead
        
        def nelder_mead_scalarize(x0, problem, bounds, ref, lo, hi, dim, max_iter=50):
            """Nelder-Mead on a scalarized objective that approximates HV contribution."""
            # Scalarize: minimize the sum of objectives weighted by proximity to ref
            # Use a Tchebycheff-style or weighted sum with adaptive weights
            # For simplicity, use weighted sum with weights based on distance to ref
            
            def scalar_obj(x):
                f = np.array(problem(x), dtype=float)
                # Weighted sum where weights favor points closer to ref in each dimension
                # Use a simple approach: minimize max(f_i / ref_i) or sum of normalized objectives
                # Actually, for HV maximization, we want to minimize the objectives
                # Use a simple weighted sum
                w = 1.0 / (ref + 1e-6)  # Normalize by ref
                return np.sum(f * w)
            
            # Nelder-Mead simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, (i - 1) % dim] += 0.1 * (hi - lo)[(i - 1) % dim]
                simplex[i] = np.clip(simplex[i], lo, hi)
            
            f_simplex = np.array([scalar_obj(s) for s in simplex])
            
            for _ in range(max_iter):
                # Sort by objective value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = scalar_obj(x_r)
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + 2 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, lo, hi)
                    f_e = scalar_obj(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = scalar_obj(x_c)
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_simplex[i] = scalar_obj(simplex[i])
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break
            
            return simplex[0], f_simplex[0]

        # Main loop: iteratively refine the most contributing point
        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Get non-dominated set
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                break
            
            # Compute marginal HV contribution for each non-dominated point
            best_idx = -1
            best_contrib = -1
            for idx in nd_idx:
                # Compute contribution of point idx
                others = np.delete(F, idx, axis=0)
                contrib = hv_contribution(F[idx], others, ref)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx == -1 or best_contrib <= 0:
                # No improvement possible, try random perturbation
                if used + 1 > budget:
                    break
                x_new = X[best_idx] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = np.array(problem(x_new), dtype=float)
                used += 1
                # Replace worst point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = x_new
                F[worst] = f_new
                continue
            
            # Refine the best point via Nelder-Mead
            x0 = X[best_idx]
            x_refined, _ = nelder_mead_scalarize(x0, problem, bounds, ref, lo, hi, dim, max_iter=30)
            
            # Evaluate the refined point
            f_refined = np.array(problem(x_refined), dtype=float)
            used += 1
            
            # Check if it improves the HV
            old_hv = hv_3d(F, ref)
            new_F = F.copy()
            new_F[best_idx] = f_refined
            new_hv = hv_3d(new_F, ref)
            
            if new_hv > old_hv:
                X[best_idx] = x_refined
                F[best_idx] = f_refined
            else:
                # Try a random perturbation
                if used + 1 > budget:
                    break
                x_new = X[best_idx] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = np.array(problem(x_new), dtype=float)
                used += 1
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = x_new
                F[worst] = f_new
        
        return X
# EVOLVE-BLOCK-END
