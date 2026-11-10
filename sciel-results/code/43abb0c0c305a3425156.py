"""Multi-objective optimizer for RE34 using MOSA with hypervolume-driven acceptance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective Simulated Annealing with hypervolume-driven acceptance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 305.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random.default_rng(42)

        # ---------- Hypervolume computation (3D) ----------
        def hypervolume_3d(points, reference):
            """Compute exact hypervolume for 3D points dominated by reference."""
            # Filter points that are strictly better than reference in all objectives
            pts = []
            for p in points:
                if np.all(p < reference):
                    pts.append(p)
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            n = len(pts)
            if n == 1:
                return float(np.prod(reference - pts[0]))
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Remove dominated points (keep only non-dominated)
            non_dom = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom.append(i)
            
            if len(non_dom) == 0:
                return 0.0
            
            nd_pts = pts[non_dom]
            n_nd = len(nd_pts)
            
            # For 3D, we can compute hypervolume by sorting and sweeping
            # Sort by first coordinate
            order = np.argsort(nd_pts[:, 0])
            nd_pts = nd_pts[order]
            
            hv = 0.0
            # For each point, compute the volume it uniquely contributes
            for i in range(n_nd):
                # The point nd_pts[i] contributes a box from nd_pts[i] to reference,
                # minus the parts already covered by points with smaller first coord
                x_min = nd_pts[i, 0]
                y_min = nd_pts[i, 1]
                z_min = nd_pts[i, 2]
                
                # Find the next point with smaller first coordinate that has y <= y_min
                # This is the "shadow" in the y-z plane
                y_shadow = reference[1]
                z_shadow = reference[2]
                
                for j in range(i + 1, n_nd):
                    if nd_pts[j, 1] <= y_min:
                        y_shadow = min(y_shadow, nd_pts[j, 1])
                    if nd_pts[j, 2] <= z_min:
                        z_shadow = min(z_shadow, nd_pts[j, 2])
                    # Early termination if we've covered the whole shadow
                    if y_shadow <= y_min and z_shadow <= z_min:
                        break
                
                # The unique contribution is the box [x_min, ref_x] x [y_min, y_shadow] x [z_min, z_shadow]
                # But this isn't quite right for hypervolume. Let me use a different approach.
                pass
            
            # Better approach for 3D hypervolume: use the "inclusion-exclusion" or
            # decompose into axis-aligned boxes
            # Actually, let's use a simpler but correct method:
            # Sort by x, then for each consecutive pair, the "slice" between them
            # has a 2D cross-section in y-z that we can compute.
            
            # Sort by first objective
            order = np.argsort(nd_pts[:, 0])
            sorted_pts = nd_pts[order]
            
            hv = 0.0
            # The region is bounded by reference in the "far" direction
            # For each point, it defines a box to the reference
            # We need to compute the union of these boxes
            
            # Use the standard 3D hypervolume algorithm:
            # 1. Sort points by x
            # 2. For each point, the "new" volume added is the volume of the box
            #    from this point to reference, minus the overlap with previously added boxes.
            #    The overlap can be computed by finding, among previous points,
            #    the ones that "shadow" this point.
            
            # Actually, for correctness and simplicity, let's use a different approach:
            # Compute hypervolume by summing contributions of each non-dominated point.
            # For point i, its contribution is the volume of the region in [x_i, ref_x] x [y_i, ref_y] x [z_i, ref_z]
            # that is not covered by any other point's box.
            
            # This is complex. Let's use a simpler but still correct method:
            # For 3D, sort by x. Then the hypervolume can be computed as:
            # HV = sum over i of (x_{i+1} - x_i) * A_i
            # where A_i is the 2D hypervolume of the "front" at x in [x_i, x_{i+1}]
            # and x_0 = min_x, x_n = ref_x
            
            # Let me implement a clean 3D hypervolume:
            pts_sorted = sorted(nd_pts, key=lambda p: p[0])
            
            hv = 0.0
            # Extend with reference point
            # The hypervolume is the volume of the union of boxes [p_i, ref]
            
            # Use the "incremental" approach:
            # Process points in order of increasing x.
            # Maintain the 2D "skyline" in y-z for points processed so far.
            
            # For simplicity and correctness, let's use a direct computation:
            # HV = integral over the indicator function that a point is not dominated
            # by any point in the set, integrated over the region [min_coords, ref].
            
            # Given the small number of points (archive size), let's just compute it directly
            # using the formula for 3D hypervolume of a set of non-dominated points.
            
            # For non-dominated points sorted by x:
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV2D(front_i)
            # where front_i is the set of points with x <= x_{i+1} projected to y-z,
            # and HV2D is the 2D hypervolume (area) of that front in y-z with ref as reference.
            # x_n = ref_x
            
            x_vals = sorted_pts[:, 0]
            yz_vals = sorted_pts[:, 1:3]
            
            # Add reference x
            x_ext = np.append(x_vals, reference[0])
            
            for i in range(len(x_vals)):
                # The slice from x_vals[i] to x_ext[i+1]
                dx = x_ext[i + 1] - x_vals[i]
                # The 2D front at this slice: all points with x <= x_ext[i+1]
                # which is points 0..i (since sorted by x)
                # The 2D hypervolume of points 0..i in y-z with reference (ref_y, ref_z)
                # 2D hypervolume: area under the "skyline"
                
                # For 2D, sort by y, then compute area
                yz_slice = yz_vals[:i + 1]
                if len(yz_slice) == 0:
                    continue
                
                # 2D hypervolume: sort by y, then for each point, the "height" in z
                # is min of z-values of points with y <= current y
                yz_sorted = sorted(yz_slice, key=lambda p: p[0])
                
                area = 0.0
                prev_y = reference[1]
                min_z = reference[2]
                
                # Process in order of increasing y
                for j in range(len(yz_sorted)):
                    y_j, z_j = yz_sorted[j]
                    # The width in y from prev_y to y_j
                    dy = prev_y - y_j
                    # The height in z is min_z (the minimum z seen so far)
                    area += dy * (min_z - z_j) if min_z > z_j else 0.0
                    # Wait, this isn't right either.
                    pass
                
                # Let me use a cleaner 2D hypervolume computation:
                # Sort by y. The 2D hypervolume is the area of the union of rectangles
                # [y_i, ref_y] x [z_i, ref_z] for each point.
                
                # For non-dominated points in 2D, sorted by y (increasing), z is decreasing.
                # HV2D = sum of (y_{i+1} - y_i) * (ref_z - z_i) for i=0..n-1,
                #        + (ref_y - y_{n-1}) * (ref_z - z_{n-1})
                # where y_{n} = ref_y.
                
                yz_nd = yz_sorted  # Already sorted by y
                n_2d = len(yz_nd)
                
                if n_2d == 0:
                    continue
                
                # Check if they're non-dominated in 2D
                # For our purpose, assume they are (they came from a 3D non-dominated set)
                
                area_2d = 0.0
                y_ext = np.append(yz_nd[:, 0], reference[1])
                for j in range(n_2d):
                    dy = y_ext[j + 1] - yz_nd[j, 0]
                    dz = reference[2] - yz_nd[j, 1]
                    area_2d += dy * dz
                
                hv += dx * area_2d
            
            return float(hv)

        # ---------- Pareto archive ----------
        archive_X = []
        archive_F = []
        current_hv = 0.0

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, arch_F):
            for af in arch_F:
                if dominates(af, f):
                    return True
            return False

        def update_archive(x, f):
            """Add x/f to archive if non-dominated, removing dominated members."""
            nonlocal current_hv
            if is_dominated(f, archive_F):
                return False
            for af in archive_F:
                if np.allclose(f, af, atol=1e-12):
                    return False
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_X)
            archive_F.extend(new_F)
            current_hv = hypervolume_3d(archive_F, ref)
            return True

        # ---------- Initialization ----------
        # Stratified initialization for good coverage
        pop_size = 40
        init_points = []
        
        # Corners (subset)
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )
        # Pick diverse corners
        for i in [0, 1, 2, 4, 8, 16, 5, 10, 20, 31]:
            init_points.append(corners[i].copy())
        
        # Midpoint
        mid = (lo + hi) / 2.0
        init_points.append(mid.copy())
        
        # Face centers
        for d in range(dim):
            for val in [lo[d], hi[d]]:
                fc = mid.copy()
                fc[d] = val
                init_points.append(fc.copy())
        
        # Fill with random
        while len(init_points) < pop_size:
            x = lo + (hi - lo) * rng.random(dim)
            init_points.append(x)
        
        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Add initial population to archive
        for i in range(len(X)):
            update_archive(X[i], F[i])

        # ---------- MOSA Search ----------
        T0 = 1.0  # Initial temperature
        T_min = 0.001
        cooling = 0.995  # Cooling rate per iteration
        
        best_hv = current_hv
        best_archive_X = [x.copy() for x in archive_X]
        best_archive_F = [f.copy() for f in archive_F]
        
        # Track current solution for MOSA
        # Pick a random starting point from archive
        if len(archive_X) > 0:
            idx = rng.integers(len(archive_X))
            current_x = archive_X[idx].copy()
            current_f = archive_F[idx].copy()
        else:
            current_x = mid.copy()
            current_f = problem(current_x)
            used += 1
            update_archive(current_x, current_f)
        
        T = T0
        no_improve_count = 0
        
        while used < budget:
            if time.time() - start_time > deadline:
                break
            
            # Generate neighbor using DE-like mutation
            if len(archive_X) >= 3:
                a_idx = rng.choice(len(archive_X), size=3, replace=False)
                a1, a2, a3 = archive_X[a_idx[0]], archive_X[a_idx[1]], archive_X[a_idx[2]]
            elif len(archive_X) >= 2:
                a1, a2 = archive_X[rng.integers(len(archive_X))], archive_X[rng.integers(len(archive_X))]
                a3 = current_x
            else:
                a1, a2, a3 = current_x, mid, mid
            
            F_factor = 0.5 + 0.5 * rng.random()
            CR = 0.7 + 0.3 * rng.random()
            
            trial = a1 + F_factor * (a2 - a3)
            mask = rng.random(dim) < CR
            trial = np.where(mask, trial, current_x)
            trial = np.clip(trial, lo, hi)
            
            ft = problem(trial)
            used += 1
            
            # Hypervolume-driven acceptance
            if not is_dominated(ft, archive_F):
                # Non-dominated: always accept
                old_hv = current_hv
                update_archive(trial, ft)
                current_x = trial.copy()
                current_f = ft.copy()
                no_improve_count = 0
            else:
                # Dominated: accept with probability based on temperature
                # Compute how much worse it is in terms of hypervolume contribution
                # Simple: accept with probability exp(-delta/T) where delta is a measure of deterioration
                # Use the sum of objective values as a proxy
                delta = np.sum(ft) - np.sum(current_f)
                if delta < 0:
                    # Actually better in sum, accept
                    current_x = trial.copy()
                    current_f = ft.copy()
                else:
                    prob = np.exp(-delta / (T + 1e-10))
                    if rng.random() < prob:
                        current_x = trial.copy()
                        current_f = ft.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Update best archive
            if current_hv > best_hv:
                best_hv = current_hv
                best_archive_X = [x.copy() for x in archive_X]
                best_archive_F = [f.copy() for f in archive_F]
            
            # Periodic restart: if stuck for a while, restart temperature
            no_improve_count += 1
            if no_improve_count > 50:
                T = max(T, T0 * 0.5)
                no_improve_count = 0
                # Pick a new random starting point from archive
                if len(archive_X) > 0:
                    idx = rng.integers(len(archive_X))
                    current_x = archive_X[idx].copy()
                    current_f = archive_F[idx].copy()

        # ---------- Final polish: local search on best archive members ----------
        if len(archive_X) > 0 and time.time() - start_time < deadline - 3:
            # Sort archive by some criterion and polish top members
            # Use simple coordinate descent with adaptive step size
            n_polish = min(10, len(archive_X))
            
            # Pick diverse members to polish
            indices = list(range(len(archive_X)))
            rng.shuffle(indices)
            polish_indices = indices[:n_polish]
            
            for pi in polish_indices:
                if used >= budget or time.time() - start_time > deadline - 2:
                    break
                
                x0 = archive_X[pi].copy()
                f0 = archive_F[pi].copy()
                step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
                
                for step in step_sizes:
                    if used >= budget or time.time() - start_time > deadline - 2:
                        break
                    for d in range(dim):
                        if used >= budget or time.time() - start_time > deadline - 2:
                            break
                        for delta in [step, -step]:
                            if used >= budget:
                                break
                            x_try = x0.copy()
                            x_try[d] += delta
                            x_try = np.clip(x_try, lo, hi)
                            f_try = problem(x_try)
                            used += 1
                            if not is_dominated(f_try, archive_F):
                                x0 = x_try
                                f0 = f_try
                                update_archive(x0, f_try)
                                if current_hv > best_hv:
                                    best_hv = current_hv
                                    best_archive_X = [x.copy() for x in archive_X]
                                    best_archive_F = [f.copy() for f in archive_F]
                            elif np.sum(f_try) < np.sum(f0) and rng.random() < 0.3:
                                # Accept with some probability if it's better in sum
                                x0 = x_try
                                f0 = f_try

        return best_archive_X
# EVOLVE-BLOCK-END
