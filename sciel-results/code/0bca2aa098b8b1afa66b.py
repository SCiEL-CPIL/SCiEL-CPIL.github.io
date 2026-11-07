# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 330.0 - 30.0  # 300s budget + 30s safety margin
        rng = np.random.default_rng(12345)

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([3000.0, 0.06])

        def hv_contribution(points):
            """Compute 2D hypervolume contribution of each point relative to reference.
            Returns array of HV contributions (hypervolume each point adds if removed).
            points: (n, 2) array of objective values.
            """
            n = len(points)
            if n == 0:
                return np.array([])
            # Filter to points strictly better than reference in both objectives
            mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
            if not np.any(mask):
                return np.zeros(n)
            pts = points[mask]
            # Sort by first objective ascending
            order = np.argsort(pts[:, 0])
            pts_s = pts[order]
            n_s = len(pts_s)
            # For 2D: contribution of point i = (ref0 - x_i) * (y_{i+1} - y_i)
            # where y is the minimum y among points with x <= x_i
            # Actually: sort by x asc, then contribution_i = (ref0 - x_i) * (min_y_suffix - y_i)
            # where min_y_suffix[i] = min(y_i, y_{i+1}, ..., y_{n-1})
            y_vals = pts_s[:, 1]
            x_vals = pts_s[:, 0]
            # min y from i to end
            min_y_suffix = np.minimum.accumulate(y_vals[::-1])[::-1]
            # contribution for each point in sorted order
            contrib_sorted = np.zeros(n_s)
            for i in range(n_s):
                # width in x: ref0 - x_i
                # height in y: min_y_suffix[i] - y_i  (but this is the "unique" y range)
                # Actually the standard 2D HV contribution:
                # Sort by x ascending. For point i, its contribution is:
                # (ref0 - x_i) * (next_y - y_i) where next_y = min(y_j for j > i) or ref1
                # But we need to be careful: the contribution is the area that point i
                # uniquely covers. In 2D, after sorting by x asc:
                # The "skyline" contribution: point i contributes (ref0 - x_i) * (y_{i-1} - y_i)
                # where y_{i-1} is the y of the previous point (with smaller x) that is
                # still above y_i... 
                
                # Simpler approach: use the standard formula.
                # Sort by x ascending. The HV is sum over consecutive points of trapezoids.
                # Contribution of point i (in sorted order) = (ref0 - x_i) * (y_{i+1} - y_i)
                # where y_{i+1} = min(y_{i+1}, ..., y_{n-1}, ref1)
                pass
            
            # Let me use a cleaner approach:
            # Sort by x ascending. For each point i in sorted order:
            #   its contribution = (ref0 - x_i) * (y_next - y_i)
            #   where y_next = min(y_{i+1}, ..., y_{n-1}, ref1)
            # This works because in 2D, the non-dominated set sorted by x has
            # decreasing y, and each point's unique area is the rectangle to its
            # right and below the next point's y.
            
            # Recompute properly
            # First, get the non-dominated subset sorted by x asc
            # A point is dominated if there exists another with x' <= x and y' <= y (strict in at least one)
            # For 2D: point i dominated by j if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i)
            
            # Let's just compute the total HV and each point's marginal contribution
            # by removal. For n points this is O(n^2) but n is small (population size).
            
            contribs = np.zeros(n_s)
            for i in range(n_s):
                # Total HV without point i
                if n_s == 1:
                    total_hv = (ref[0] - x_vals[0]) * (ref[1] - y_vals[0])
                    total_hv_wo = 0.0
                else:
                    # Compute HV of all points except i
                    mask_wo = np.ones(n_s, dtype=bool)
                    mask_wo[i] = False
                    pts_wo = pts_s[mask_wo]
                    # Sort by x
                    order_wo = np.argsort(pts_wo[:, 0])
                    pts_wo_s = pts_wo[order_wo]
                    # 2D HV: sort by x asc, integrate
                    x_w = pts_wo_s[:, 0]
                    y_w = pts_wo_s[:, 1]
                    # The HV is the area under the "skyline" from left to right
                    # Sort by x asc, then for each segment, the height is min y from that point onward
                    min_y_from_right = np.minimum.accumulate(y_w[::-1])[::-1]
                    # x boundaries: x_0, x_1, ..., x_{m-1}, ref0
                    # For segment from x_j to x_{j+1} (or ref0), height = min_y_from_right[j]
                    # But we need to be careful: the skyline is the lower envelope
                    # Actually for minimization, the Pareto front has x increasing, y decreasing
                    # The HV is sum of rectangles: for each point j in sorted order,
                    # width = (x_{j+1} - x_j) or (ref0 - x_j) for last
                    # height = y_j - (y_{j+1} if j+1 exists else ref1)
                    # Wait, that's not right either.
                    
                    # Standard 2D HV computation:
                    # Sort points by x ascending.
                    # The HV = sum_{j=0}^{m-1} (x_{j+1} - x_j) * (y_j - ref1)  ... no
                    # 
                    # Let me think again. For minimization with reference (ref0, ref1):
                    # A point (x, y) is better than ref if x < ref0 and y < ref1.
                    # The HV is the volume of the region dominated by the set and 
                    # containing ref.
                    # In 2D: HV = area of union of rectangles [x_j, ref0] x [y_j, ref1]
                    # 
                    # To compute: sort by x ascending. The skyline has x increasing, y decreasing.
                    # HV = sum over j of (x_{j+1} - x_j) * (y_j - ref1) for j < m-1
                    #     + (ref0 - x_{m-1}) * (y_{m-1} - ref1)
                    # where x_{m} = ref0
                    # But this only works if the points form a proper skyline (x inc, y dec).
                    # We need to first extract the non-dominated set.
                    
                    # For a set of points, the non-dominated set in 2D:
                    # Sort by x asc. Then a point is non-dominated if its y is less than
                    # all y's of points to its left (with smaller x).
                    # Actually: sort by x asc, then y asc (for ties). 
                    # A point is dominated if there's a point with x' <= x and y' <= y.
                    # After sorting by x asc, for each point, it's non-dominated if 
                    # y < min(y of all previous points).
                    
                    # This is getting complex. Let me use a simpler O(n^2) approach
                    # for marginal HV contribution since population is small.
                    pass
                
                # Use O(n^2) marginal HV: total_hv - total_hv_without_i
                # But computing total_hv for each subset is O(n^2 log n) total.
                # For population size ~50-100, this is fine.
                pass
            
            # Let me restart with a clean, correct implementation
            return contribs  # placeholder

        # Clean implementation of 2D hypervolume and marginal contributions
        def total_hv_2d(pts):
            """Compute 2D hypervolume of a set of points relative to reference.
            pts: (n, 2) array. Returns scalar HV.
            """
            if len(pts) == 0:
                return 0.0
            mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by x ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            x = pts[:, 0]
            y = pts[:, 1]
            n = len(pts)
            # Find the non-dominated subset: 
            # After sorting by x asc, a point is non-dominated if y < min(y[0..i-1])
            # Build skyline
            skyline_y = np.full(n, np.inf)
            min_y_so_far = np.inf
            for i in range(n):
                if y[i] < min_y_so_far:
                    min_y_so_far = y[i]
                    skyline_y[i] = y[i]
                else:
                    skyline_y[i] = np.nan  # dominated
            # Get non-dominated indices
            nd_idx = np.where(~np.isnan(skyline_y))[0]
            if len(nd_idx) == 0:
                return 0.0
            x_nd = x[nd_idx]
            y_nd = skyline_y[nd_idx]
            # Compute HV: sum of rectangles
            # For each consecutive pair in skyline, plus last to ref
            hv = 0.0
            for j in range(len(x_nd)):
                x_next = x_nd[j+1] if j+1 < len(x_nd) else ref[0]
                width = x_next - x_nd[j]
                height = y_nd[j] - ref[1]
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def marginal_hv_2d(pts):
            """Compute marginal HV contribution of each point.
            pts: (n, 2). Returns (n,) array of contributions.
            """
            n = len(pts)
            if n == 0:
                return np.array([])
            mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
            if not np.any(mask):
                return np.zeros(n)
            total = total_hv_2d(pts)
            contribs = np.zeros(n)
            for i in range(n):
                if not mask[i]:
                    contribs[i] = 0.0
                    continue
                pts_wo = np.delete(pts, i, axis=0)
                hv_wo = total_hv_2d(pts_wo)
                contribs[i] = total - hv_wo
            return contribs

        # Non-dominated sort for selection
        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F (n, n_obj)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                # Check if i is dominated by any other point
                # j dominates i if F[j] <= F[i] and F[j] != F[i]
                diff = F - F[i]  # (n, n_obj), diff[k] = F[k] - F[i]
                # j dominates i if all diff[j] <= 0 and any diff[j] < 0
                dominated_by = np.any((diff <= 0).all(axis=1) & (diff < 0).any(axis=1))
                # Remove self
                dominated_by[i] = False
                if dominated_by.any():
                    nd[i] = False
            return nd

        # Initialize population
        # Reference-point-aligned grid + random
        pop_size = min(50, max(10, budget // 4))
        X = np.zeros((pop_size, dim))
        
        # Mix of strategies for initialization
        # 1/3: random in full box
        # 1/3: random near reference-aligned region (lower objectives)
        # 1/3: stratified grid
        
        n_rand = pop_size // 3
        n_near = pop_size // 3
        n_grid = pop_size - n_rand - n_near
        
        idx = 0
        # Random in full box
        for _ in range(n_rand):
            X[idx] = lo + span * rng.random(dim)
            idx += 1
        # Random in lower half (bias toward better objectives)
        for _ in range(n_near):
            X[idx] = lo + span * rng.random(dim) * 0.5
            idx += 1
        # Stratified grid
        if n_grid > 0:
            # Simple LHS-like or grid
            n_per_dim = max(1, int(round(n_grid ** (1.0 / dim))))
            grid_pts = []
            for d in range(dim):
                grid_pts.append(np.linspace(lo[d], hi[d], n_per_dim + 1))
            # Take a subset of grid points
            from itertools import product
            all_grid = list(product(*[np.linspace(lo[d], hi[d], n_per_dim) for d in range(dim)]))
            rng.shuffle(all_grid)
            for _ in range(n_grid):
                X[idx] = all_grid[_ % len(all_grid)]
                idx += 1
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive differential evolution loop
        F_rate = 0.8
        CR = 0.5
        
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Get non-dominated parents
            nd_mask = non_dominated_mask(F)
            if not np.any(nd_mask):
                nd_mask = np.ones(len(F), dtype=bool)
            parents = X[nd_mask]
            F_parents = F[nd_mask]
            
            # Select 3 distinct parents for DE
            n_p = len(parents)
            if n_p >= 3:
                i1, i2, i3 = rng.choice(n_p, size=3, replace=False)
                mutant = parents[i1] + F_rate * (parents[i2] - parents[i3])
            else:
                # Fallback: random mutation
                i1 = rng.integers(n_p)
                mutant = parents[i1] + 0.5 * span * rng.standard_normal(dim)
            
            # Crossover
            child = mutant.copy()
            for d in range(dim):
                if rng.random() < CR:
                    pass  # keep mutant
                else:
                    # Take from a random parent
                    ip = rng.integers(n_p)
                    child[d] = parents[ip][d]
            
            # Ensure at least one gene from mutant
            if not np.any(child != mutant):
                d0 = rng.integers(dim)
                child[d0] = mutant[d0]
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace worst point by HV contribution
            contribs = marginal_hv_2d(F)
            worst = int(np.argmin(contribs))
            X[worst] = child
            F[worst] = fc
        
        return X
# EVOLVE-BLOCK-END
