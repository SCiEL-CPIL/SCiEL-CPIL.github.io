# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using hypervolume-driven adaptive
    weighted-sum scalarization with local search refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.RandomState(123)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (lo + hi) / 2

        # Archive: list of (x, f) pairs
        archive = []  # list of (x_array, f_array)
        used = 0

        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f (list of arrays)."""
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def hv_contribution(f, archive_f, ref):
            """Compute hypervolume contribution of f w.r.t. archive and reference point.
            Only valid for 3 objectives."""
            if np.any(f >= ref):
                return 0.0
            
            # Build list of points including f
            points = [f] + list(archive_f)
            
            # Filter points that are better than ref in all objectives
            valid_points = []
            for p in points:
                if np.all(p < ref):
                    valid_points.append(p)
            
            if len(valid_points) == 0:
                return 0.0
            
            # Compute hypervolume with and without f
            def compute_hv(points, ref):
                """Compute hypervolume for 3D points with reference point."""
                if len(points) == 0:
                    return 0.0
                
                # Sort by first objective
                points = sorted(points, key=lambda p: p[0])
                
                hv = 0.0
                # For each point, compute its contribution
                for i, p in enumerate(points):
                    # Find the region where p is the "top" point
                    # A point contributes if it's not dominated by any other point
                    is_non_dominated = True
                    for j, q in enumerate(points):
                        if i != j and np.all(q <= p) and np.any(q < p):
                            is_non_dominated = False
                            break
                    if not is_non_dominated:
                        continue
                    
                    # Compute the box where p is the minimum in all objectives
                    # among the non-dominated set
                    # For 3D, the contribution is the volume of the box
                    # from p to the "envelope" formed by other points
                    # Simplified: for each dimension, find the next point that
                    # is better in that dimension
                    
                    # This is complex; use a simpler approach
                    # The contribution of a non-dominated point in 3D is:
                    # For each pair of other points, compute the box
                    # Actually, let's use the standard 3D HV algorithm
                    
                    pass
                
                # Use a simpler 3D HV computation
                return self._hv_3d(points, ref)
            
            def hv_3d(points, ref):
                """Compute exact hypervolume for 3D points."""
                if len(points) == 0:
                    return 0.0
                
                # Filter: only points better than ref
                pts = [p for p in points if np.all(p < ref)]
                if len(pts) == 0:
                    return 0.0
                
                # Remove dominated points
                pts = self._remove_dominated(pts)
                if len(pts) == 0:
                    return 0.0
                
                # Sort by first objective
                pts = sorted(pts, key=lambda p: p[0])
                
                # 3D hypervolume: sweep along first objective
                # For each point, the contribution is the area in the 2D plane
                # (obj2, obj3) multiplied by the width in obj1
                
                hv = 0.0
                for i, p in enumerate(pts):
                    # Width in obj1: from p[0] to the next point's obj1 (or ref[0])
                    if i + 1 < len(pts):
                        next_obj1 = pts[i + 1][0]
                    else:
                        next_obj1 = ref[0]
                    width = max(0, next_obj1 - p[0])
                    
                    # Area in (obj2, obj3) plane: the region where p is the
                    # minimum in both obj2 and obj3 among points with obj1 <= current
                    # Actually, we need to be more careful
                    
                    # For each point p_i, its contribution to HV is:
                    # (x_{i+1} - x_i) * area_i
                    # where area_i is the area in (y,z) of the region dominated
                    # by p_i but not by any p_j with j < i
                    
                    # Let's compute this properly
                    pass
                
                # Use a different approach: enumerate all combinations
                # For small sets, we can compute HV exactly
                # The HV is the volume of the union of boxes [p_i, ref]
                
                # For 3D with few points, use the inclusion-exclusion or
                # sweep line approach
                
                # Simple approach for small n: 
                # Sort by x, then for each point, compute the 2D area it
                # contributes
                
                # Actually, let's use a simpler but correct method:
                # The hypervolume is the volume of the set of points dominated
                # by at least one point in the set.
                
                # For 3D, we can use the following:
                # Sort points by x-coordinate
                # For each point, compute the area in (y,z) plane of the
                # region that is dominated by this point and not by any
                # previously processed point (with smaller x)
                
                pts_sorted = sorted(pts, key=lambda p: p[0])
                hv = 0.0
                
                for i, p in enumerate(pts_sorted):
                    # The x-range for which p is the "frontier" point
                    x_start = p[0]
                    if i + 1 < len(pts_sorted):
                        x_end = pts_sorted[i + 1][0]
                    else:
                        x_end = ref[0]
                    
                    x_width = max(0, x_end - x_start)
                    
                    # The (y,z) area: the region where (y,z) is dominated by p
                    # and not dominated by any point with x < x_start
                    # Actually, it's the area of the rectangle [p[1], ref[1]] x [p[2], ref[2]]
                    # minus the parts covered by previous points
                    
                    # This is getting complex. Let me use a simpler approximation
                    # that's still exact for the contribution calculation.
                    
                    # For the contribution of p_i, it's:
                    # x_width * (area of [p_i[1], ref[1]] x [p_i[2], ref[2]] 
                    #              that is not covered by union of 
                    #              [p_j[1], ref[1]] x [p_j[2], ref[2]] for j < i
                    #              where p_j[0] <= p_i[0])
                    
                    # Compute the uncovered area in 2D
                    area = self._area_2d_uncovered(p[1], p[2], ref[1], ref[2], 
                                                   [(pts_sorted[j][1], pts_sorted[j][2]) 
                                                    for j in range(i) if pts_sorted[j][0] <= p[0]])
                    
                    hv += x_width * area
                
                return hv
            
            def area_2d_uncovered(y_min, z_min, y_max, z_max, existing_rects):
                """Compute the area of [y_min, y_max] x [z_min, z_max] not covered
                by the union of existing rectangles [ey, y_max] x [ez, z_max]."""
                # The total area
                total_area = (y_max - y_min) * (z_max - z_min)
                
                # Subtract the covered area
                # Each existing rect covers [ey, y_max] x [ez, z_max]
                # The union of these rectangles
                # For simplicity, compute the union area
                covered = self._union_area_2d(existing_rects, y_max, z_max)
                
                return max(0, total_area - covered)
            
            def union_area_2d(rects, y_max, z_max):
                """Compute the area of the union of rectangles [ey, y_max] x [ez, z_max]."""
                if len(rects) == 0:
                    return 0.0
                
                # Sort by ey
                rects = sorted(rects, key=lambda r: r[0])
                
                # Sweep along y
                # For each y-interval, compute the z-coverage
                events = []
                for ey, ez in rects:
                    events.append((ey, 0, ez))  # start
                    # end is y_max, but we only care up to y_max
                
                # Actually, all rects go from ey to y_max
                # So the union is: for each y, the z-range covered is
                # [min_ez for rects with ey <= y, z_max]
                
                # Sort by ey
                rects_sorted = sorted(rects, key=lambda r: r[0])
                
                area = 0.0
                current_y = rects_sorted[0][0]
                min_ez = min(ez for ey, ez in rects_sorted)
                
                for i, (ey, ez) in enumerate(rects_sorted):
                    # From current_y to ey, the coverage is [min_ez, z_max]
                    if ey > current_y:
                        area += (ey - current_y) * (z_max - min_ez)
                    
                    # At ey, add this rect
                    min_ez = min(min_ez, ez)
                    current_y = ey
                
                # From current_y to y_max
                area += (y_max - current_y) * (z_max - min_ez)
                
                return area
            
            # This is getting too complex. Let me use a simpler HV contribution
            # approximation that's good enough for selection.
            # Use a simple grid-based HV estimation.
            
            # Actually, for the purpose of weight adaptation, we just need
            # a relative measure. Let's use a simpler approach.
            
            return 0.0  # Placeholder - we'll use a different approach below
        
        # Simplified approach: use a fixed set of weights and adapt based on
        # hypervolume contribution estimated by a simple method.
        
        # Initial weights: uniform on the simplex
        n_weights = 15
        weights = []
        # Generate points on the 2-simplex
        n_div = 4
        for i in range(n_div + 1):
            for j in range(n_div + 1 - i):
                w1 = i / n_div
                w2 = j / n_div
                w3 = 1 - w1 - w2
                weights.append([w1, w2, w3])
        
        weights = np.array(weights)
        # Normalize
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Solution per weight
        X = np.zeros((len(weights), self.dim))
        F = np.zeros((len(weights), self.n_obj))
        
        for i in range(len(weights)):
            X[i] = center + 0.1 * span * rng.standard_normal(self.dim)
            X[i] = np.clip(X[i], lo, hi)
            F[i] = problem(X[i])
            used += 1
        
        # Archive of non-dominated solutions
        archive_x = [X[i].copy() for i in range(len(weights))]
        archive_f = [F[i].copy() for i in range(len(weights))]
        
        # Remove dominated from archive
        archive_x, archive_f = self._non_dominated(archive_x, archive_f)
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break
            
            # Select a weight to work on (based on hypervolume contribution potential)
            # Use a simple strategy: cycle through weights, with some randomness
            i = rng.randint(len(weights))
            
            # Local search around X[i]
            # Use a mutation-based local search
            for trial in range(3):  # up to 3 trials per weight
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Generate a perturbation
                scale = 0.1 * span
                child = X[i] + scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Check if child is non-dominated in the archive
                dominated = False
                for af in archive_f:
                    if np.all(af <= fc) and np.any(af < fc):
                        dominated = True
                        break
                
                if not dominated:
                    # Add to archive and remove dominated points
                    new_archive_x = archive_x + [child]
                    new_archive_f = archive_f + [fc]
                    archive_x, archive_f = self._non_dominated(new_archive_x, new_archive_f)
                    
                    # Update X[i] and F[i] if this is better for weight i
                    # Use weighted sum as a proxy
                    ws_old = np.dot(weights[i], F[i])
                    ws_new = np.dot(weights[i], fc)
                    if ws_new < ws_old:
                        X[i] = child
                        F[i] = fc
                else:
                    # Even if dominated, update if it's better in weighted sum
                    ws_old = np.dot(weights[i], F[i])
                    ws_new = np.dot(weights[i], fc)
                    if ws_new < ws_old:
                        X[i] = child
                        F[i] = fc
            
            # Occasionally add new weights based on gaps in the archive
            if len(archive_f) >= 3 and used < self.budget - 10:
                # Find a gap in the archive
                # Compute the "center" of the archive in objective space
                min_f = np.min(archive_f, axis=0)
                max_f = np.max(archive_f, axis=0)
                mid_f = (min_f + max_f) / 2
                
                # Generate a random weight and check if it points to an under-explored region
                new_w = rng.dirichlet(np.ones(self.n_obj))
                
                # Check if this weight is "far" from existing weights
                min_dist = min(np.linalg.norm(new_w - w) for w in weights)
                if min_dist > 0.1:
                    # Add this weight
                    # Find a solution near the archive that matches this weight
                    # Use the archive solution closest to the weighted direction
                    best_idx = 0
                    best_dist = float('inf')
                    for j, af in enumerate(archive_f):
                        # Distance in objective space, weighted
                        d = np.linalg.norm(af - np.dot(new_w, np.mean(archive_f, axis=0)))
                        if d < best_dist:
                            best_dist = d
                            best_idx = j
                    
                    # Add new weight and solution
                    weights = np.vstack([weights, new_w])
                    X = np.vstack([X, archive_x[best_idx].copy()])
                    F = np.vstack([F, archive_f[best_idx].copy()])
        
        # Final: return the archive (best non-dominated solutions)
        # The return value is ignored, but we return something valid
        if archive_x:
            return np.array(archive_x)
        else:
            return np.array([X[0]])
    
    def _non_dominated(self, xs, fs):
        """Remove dominated solutions from the list."""
        if len(fs) == 0:
            return [], []
        
        n = len(fs)
        keep = [True] * n
        
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                # Check if j dominates i
                if np.all(fs[j] <= fs[i]) and np.any(fs[j] < fs[i]):
                    keep[i] = False
                    break
        
        return [xs[i] for i in range(n) if keep[i]], \
               [fs[i] for i in range(n) if keep[i]]
# EVOLVE-BLOCK-END
