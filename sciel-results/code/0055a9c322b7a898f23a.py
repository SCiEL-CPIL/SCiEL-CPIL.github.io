"""Multi-objective optimizer for WFG4 with hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin from 330s limit
        
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size for NSGA-II
        pop_size = 40
        # Reserve some budget for final local optimization
        polish_budget = 30
        global_budget = self.budget - polish_budget
        
        # Initial population: mix of random and structured sampling
        X = np.zeros((pop_size, self.dim))
        # First few points: structured (LHS-like)
        n_struct = min(10, pop_size)
        for i in range(n_struct):
            for d in range(self.dim):
                # Space out in each dimension
                X[i, d] = lo[d] + range_[d] * ((i + 0.5) / n_struct + rng.random() * 0.1)
        # Rest: random
        for i in range(n_struct, pop_size):
            X[i] = lo + range_ * rng.random(self.dim)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume (from task: [3.0, 5.0, 7.0])
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def compute_hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points using inclusion-exclusion."""
            # Filter points that are strictly better than reference in all objectives
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            
            pts = points[valid]
            n = len(pts)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref - pts[0]))
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Use the "sweep" approach for 3D hypervolume
            # For each point, compute the volume of the box it uniquely contributes
            # This is O(n^2) but n is small (<= pop_size)
            
            # Alternative: use a recursive decomposition
            # Sort by x, then for each point, the hypervolume it contributes is:
            # (ref_x - x_i) * (area in yz-plane of the region dominated by this point 
            # but not by any point with smaller x)
            
            # Simpler approach for small n: enumerate all subsets? No, too expensive.
            # Use the standard 3D hypervolume algorithm:
            # 1. Sort points by x-coordinate
            # 2. Sweep from left to right, maintaining the "skyline" in yz
            # 3. For each point, compute the yz-area it adds to the skyline
            
            # Even simpler for n <= 40: use the "inclusion-exclusion" via sorting
            # Actually, let's use a different approach: compute hypervolume by
            # considering each point's contribution as the box from the point to ref,
            # minus overlaps. But that's complex.
            
            # For n <= 40, let's use a simple but correct method:
            # The hypervolume is the volume of the union of boxes [p_i, ref]
            # We can compute this by sorting and using a 2D skyline
            
            # Sort by x
            pts = pts[np.argsort(pts[:, 0])]
            
            # For each point in order, compute the yz-area that is dominated by this point
            # but not by any previously processed point (with smaller x)
            # The yz-area is determined by the "skyline" of previously processed points
            
            hv = 0.0
            # Maintain the skyline in yz-plane: a set of points such that no point dominates another
            # When we add a new point, we update the skyline
            
            # Actually, let's use a simpler method: for each point i, the contribution is
            # (ref_x - x_i) * A_i, where A_i is the area in yz-plane of the region
            # dominated by point i but not by any point j with x_j < x_i.
            
            # This requires computing 2D hypervolume for subsets, which is O(n^2) per point.
            # Total O(n^3) which is fine for n=40.
            
            for i in range(n):
                # Points that could dominate i in yz: those with x <= x_i
                # But we've sorted by x, so points 0..i have x <= x_i
                # The yz-region dominated by i but not by any of 0..i-1
                # is the 2D hypervolume of point i in yz, minus the union of boxes
                # from points 0..i-1 that have y < y_i and z < z_i
                
                # 2D hypervolume: for a set of points in 2D, the hypervolume is
                # sum over each point of (ref_y - y_j) * (ref_z - z_j) minus overlaps
                # For 2D, we can compute it by sorting one coordinate
                
                # The area dominated by point i in yz:
                # A_i = (ref_y - y_i) * (ref_z - z_i)
                # But we need to subtract the area already covered by points 0..i-1
                # that have y < y_i and z < z_i
                
                # Let's collect all points with index <= i (including i) that have
                # y < y_i and z < z_i... wait, this is getting complicated.
                
                # Let me use a cleaner approach:
                # The 3D hypervolume can be computed by:
                # 1. Sort points by x
                # 2. For each consecutive pair in x, compute the 2D hypervolume of the
                #    "active" set (points with x between the two values) in the yz-plane
                #    and multiply by the x-difference
                
                # This is the standard sweep-line approach.
                pass
            
            # Let me just use a simple but potentially slower method for n <= 40:
            # Compute hypervolume by recursive decomposition or by brute force for small n.
            
            # Actually, for n <= 40, let's use the following:
            # Hypervolume = sum over all points of the volume of the box they uniquely contribute.
            # A point uniquely contributes the box [p_i, ref] minus the union of boxes
            # [p_j, ref] for all j != i. But this requires computing 3D union volume.
            
            # Let me try a different strategy: since n is small, use the "inclusion-exclusion"
            # formula for the union of boxes. But that's 2^n subsets, too expensive.
            
            # OK, let me implement the sweep-line properly.
            
            # Sort by x
            pts_sorted = pts[np.argsort(pts[:, 0])]
            n_pts = len(pts_sorted)
            
            # The hypervolume is the integral over x of the 2D hypervolume at that x.
            # Between consecutive x-values, the set of active points is constant.
            
            # Build a list of x-intervals and the active points in each
            # x-values: pts_sorted[:, 0] and ref[0]
            x_vals = np.concatenate([pts_sorted[:, 0], [ref[0]]])
            
            hv = 0.0
            for i in range(n_pts):
                x_left = pts_sorted[i, 0]
                x_right = x_vals[i + 1]
                if x_right <= x_left:
                    continue
                dx = x_right - x_left
                # Active points: those with x <= x_right (which is all points 0..i since sorted)
                # But we need points that are "active" in the sense that their box extends to ref
                # All points 0..i have x <= x_right, so they're all active
                active = pts_sorted[:i+1]
                # 2D hypervolume of active points in yz-plane with reference (ref[1], ref[2])
                area_2d = self._hv_2d(active[:, 1], active[:, 2], ref[1], ref[2])
                hv += dx * area_2d
            
            return hv
        
        def _hv_2d(self, y, z, ref_y, ref_z):
            """Compute 2D hypervolume (area) for points (y, z) with reference (ref_y, ref_z)."""
            # Filter valid points
            valid = (y < ref_y) & (z < ref_z)
            if not np.any(valid):
                return 0.0
            y = y[valid]
            z = z[valid]
            n = len(y)
            if n == 0:
                return 0.0
            if n == 1:
                return float((ref_y - y[0]) * (ref_z - z[0]))
            
            # Sort by y
            order = np.argsort(y)
            y = y[order]
            z = z[order]
            
            # Sweep from left to right
            # At each point, the area it contributes is (ref_y - y_i) * (min_z - z_i)
            # where min_z is the minimum z among all points with y >= y_i... 
            # No, the 2D hypervolume is the area of the union of rectangles.
            
            # Standard 2D hypervolume: sort by one coord, sweep
            # The area is sum over i of (y_{i+1} - y_i) * (ref_z - min_z_active)
            # where y_{n} = ref_y and min_z_active is the min z of points with y <= current y
            
            # Actually, let me think again. The 2D hypervolume with reference (ref_y, ref_z)
            # is the area of the union of rectangles [y_i, ref_y] x [z_i, ref_z].
            
            # Sort by y
            # Sweep from left to right:
            # At each step, the "active" set is all points seen so far.
            # The contribution of the interval [y_i, y_{i+1}] is (y_{i+1} - y_i) * (ref_z - min_z_so_far)
            # where min_z_so_far is the minimum z among all points with y <= y_i.
            
            # But we need to be careful: the min_z should be the min z among points that are
            # "active" in the sense that their rectangle extends to the right.
            
            # Let me use a different approach: the 2D hypervolume can be computed as:
            # Sum over each point of (ref_y - y_i) * (ref_z - z_i) minus the overlaps.
            # For 2D, the overlap between two rectangles can be computed, but for n points it's complex.
            
            # Simpler: use the fact that the 2D hypervolume is the area under the "skyline"
            # Sort by y, and for each point, the skyline z-value is the min z among all points
            # with y >= current y... no, that's not right either.
            
            # Let me just use a simple O(n^2) method:
            # The 2D hypervolume is the area of the union of n rectangles.
            # We can compute this by sweeping and maintaining the minimum z.
            
            # Sort by y
            order = np.argsort(y)
            y_sorted = y[order]
            z_sorted = z[order]
            
            # Add the reference point as the last "point"
            y_ext = np.concatenate([y_sorted, [ref_y]])
            z_ext = np.concatenate([z_sorted, [ref_z]])
            
            # Sweep from left to right
            # At each interval [y_ext[i], y_ext[i+1]], the height is (ref_z - min_z_active)
            # where min_z_active is the min z among points with y <= y_ext[i]
            
            area = 0.0
            min_z = ref_z  # Start with reference z
            for i in range(len(y_sorted)):
                # Update min_z with current point's z
                min_z = min(min_z, z_sorted[i])
                # Interval from y_sorted[i] to y_ext[i+1]
                # But wait, we need to process in order of y
                # The interval [y_sorted[i], y_sorted[i+1]] has height (ref_z - min_z_after_i)
                # where min_z_after_i is the min z among points 0..i
                if i < len(y_sorted) - 1:
                    dx = y_sorted[i+1] - y_sorted[i]
                else:
                    dx = ref_y - y_sorted[i]
                area += dx * (ref_z - min_z)
            
            return area
        
        def fast_non_dominated_sort(F):
            """NSGA-II non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            # S[i] = set of indices dominated by i
            S = [[] for _ in range(n)]
            # n_i = number of points that dominate i
            n_dom = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[j]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            # First front: points with n_dom == 0
            front = [i for i in range(n) if n_dom[i] == 0]
            fronts = []
            front_num = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distance for points in indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            F_sub = F[indices]
            n_obj = F_sub.shape[1]
            
            for k in range(n_obj):
                f_k = F_sub[:, k]
                min_f, max_f = np.min(f_k), np.max(f_k)
                if max_f == min_f:
                    continue
                sorted_idx = np.argsort(f_k)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[i]] += (f_k[sorted_idx[i+1]] - f_k[sorted_idx[i-1]]) / (max_f - min_f)
            return dist
        
        def tournament_select(F, fronts, pop_size, n_select, rng):
            """Tournament selection based on rank and crowding distance."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Compute crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_vals[i]
            
            selected = []
            for _ in range(n_select):
                # Tournament: pick 2 random, choose better one
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    # Same rank, use crowding distance
                    if cd[i] > cd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected
        
        def crossover(x1, x2, rng):
            """SBX crossover."""
            eta = 20.0
            x1 = x1.copy()
            x2 = x2.copy()
            for d in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2*u)**(1/(eta+1))
                    else:
                        beta = (1/(2*(1-u)))**(1/(eta+1))
                    x1[d] = 0.5*((1+beta)*x1[d] + (1-beta)*x2[d])
                    x2[d] = 0.5*((1-beta)*x1[d] + (1+beta)*x2[d])
            return np.clip(x1, lo, hi), np.clip(x2, lo, hi)
        
        def mutate(x, sigma, rng):
            """Polynomial mutation."""
            eta = 20.0
            x = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.9:  # Mutation rate
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2*u)**(1/(eta+1)) - 1
                    else:
                        delta = 1 - (2*(1-u))**(1/(eta+1))
                    x[d] += delta * sigma * range_[d]
            return np.clip(x, lo, hi)
        
        # Main NSGA-II loop
        while used < global_budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)
            
            # Selection
            n_select = pop_size // 2
            parents = tournament_select(F, fronts, pop_size, n_select, rng)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            for _ in range(n_offspring // 2):
                if len(parents) < 2:
                    break
                p1 = parents[rng.integers(len(parents))]
                p2 = parents[rng.integers(len(parents))]
                c1, c2 = crossover(X[p1], X[p2], rng)
                # Mutate
                sigma = 0.1
                c1 = mutate(c1, sigma, rng)
                c2 = mutate(c2, sigma, rng)
                offspring_X.append(c1)
                offspring_X.append(c2)
                if used + 2 > global_budget:
                    break
            
            # Evaluate offspring
            for ox in offspring_X:
                if used >= global_budget:
                    break
                of = problem(ox)
                used += 1
                offspring_F.append(of)
            
            if not offspring_F:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            all_X = np.vstack([X, np.array(offspring_X)])
            all_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(all_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need to truncate this front
                    n_needed = pop_size - len(new_X)
                    cd_vals = crowding_distance(all_F, front)
                    # Sort by crowding distance, take top n_needed
                    order = np.argsort(cd_vals)[::-1]
                    for i in range(n_needed):
                        idx = front[order[i]]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        # Polish phase: local optimization on best points
        if used < self.budget:
            # Get non-dominated points
            fronts = fast_non_dominated_sort(F)
            if fronts and len(fronts[0]) > 0:
                # Take up to 6 best non-dominated points
                n_polish = min(6, len(fronts[0]))
                polish_per_point = (self.budget - used) // n_polish
                if polish_per_point > 0:
                    for p_idx in fronts[0][:n_polish]:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        x_curr = X[p_idx].copy()
                        f_curr = F[p_idx].copy()
                        
                        # Coordinate descent with adaptive step
                        step_size = 0.05
                        for iteration in range(polish_per_point):
                            if used >= self.budget or time.time() > deadline:
                                break
                            improved = False
                            for d in range(self.dim):
                                if used >= self.budget or time.time() > deadline:
                                    break
                                # Try several step sizes
                                for step in [step_size, step_size * 0.5, step_size * 2]:
                                    if used >= self.budget:
                                        break
                                    for sign in [1, -1]:
                                        if used >= self.budget:
                                            break
                                        x_trial = x_curr.copy()
                                        x_trial[d] += sign * step * range_[d]
                                        x_trial = np.clip(x_trial, lo, hi)
                                        
                                        f_trial = problem(x_trial)
                                        used += 1
                                        
                                        # Accept if it dominates or improves sum
                                        if np.all(f_trial <= f_curr) and np.any(f_trial < f_curr):
                                            x_curr = x_trial
                                            f_curr = f_trial
                                            improved = True
                                            step_size = step * 0.5
                                            break
                                        elif np.sum(f_trial) < np.sum(f_curr) - 0.01:
                                            x_curr = x_trial
                                            f_curr = f_trial
                                            improved = True
                                            step_size = step * 0.5
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if not improved:
                                step_size *= 0.5
                                if step_size < 0.005:
                                    break
                        
                        X[p_idx] = x_curr
                        F[p_idx] = f_curr
        
        return X
# EVOLVE-BLOCK-END
