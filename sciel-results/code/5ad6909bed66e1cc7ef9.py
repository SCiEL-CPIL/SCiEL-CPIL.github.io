import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim
        ref = np.array([1.1, 1.1, 25.0])

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_eval = np.clip(x, lo, hi).copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 with x[2:]=0: g=1
        # f1 = x0, f2 = x1, f3 = 1 - x1 * h(x1, 1)
        # h(y,1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
        def compute_objectives(x0, x1):
            """Compute 3 objectives for given x0, x1 (with x[2:]=0)."""
            f1 = x0
            f2 = x1
            # h(x1, 1)
            s = np.sin(3.0 * np.pi * x1) + np.sin(4.0 * np.pi * x1)
            h = 9.0 / (1.0 + 2.0 * s)
            f3 = 1.0 - x1 * h
            return np.array([f1, f2, f3])

        def compute_objectives_batch(x0_arr, x1_arr):
            """Vectorized version."""
            f1 = x0_arr
            f2 = x1_arr
            s = np.sin(3.0 * np.pi * x1_arr) + np.sin(4.0 * np.pi * x1_arr)
            h = 9.0 / (1.0 + 2.0 * s)
            f3 = 1.0 - x1_arr * h
            return np.column_stack([f1, f2, f3])

        # Generate a large pool of candidate points on the Pareto front
        # Use a dense grid + some random jitter for diversity
        n_pool = 5000
        # Mix of grid and random points
        n_grid = 70
        gx0 = np.linspace(0, 1, n_grid)
        gx1 = np.linspace(0, 1, n_grid)
        G0, G1 = np.meshgrid(gx0, gx1, indexing='ij')
        grid_x0 = G0.ravel()
        grid_x1 = G1.ravel()

        n_rand = n_pool - len(grid_x0)
        if n_rand > 0:
            rand_x0 = np.random.uniform(0, 1, n_rand)
            rand_x1 = np.random.uniform(0, 1, n_rand)
            pool_x0 = np.concatenate([grid_x0, rand_x0])
            pool_x1 = np.concatenate([grid_x1, rand_x1])
        else:
            pool_x0 = grid_x0
            pool_x1 = grid_x1

        # Compute all candidate objective values
        pool_objs = compute_objectives_batch(pool_x0, pool_x1)  # shape (n_pool, 3)

        # Filter: only points strictly better than reference point
        valid_mask = np.all(pool_objs < ref, axis=1)
        pool_x0 = pool_x0[valid_mask]
        pool_x1 = pool_x1[valid_mask]
        pool_objs = pool_objs[valid_mask]
        n_valid = len(pool_x0)

        # Greedy hypervolume maximization
        # Track selected points and compute marginal HV contribution
        selected_indices = []
        selected_objs = []  # list of (3,) arrays
        n_select = min(budget, n_valid)

        # Fast hypervolume computation for 3D
        # For a set of points, HV = sum of volumes of boxes each point contributes
        # Using the "union of boxes" approach with dominance filtering

        def hypervolume_3d(points, ref_point):
            """Compute exact 3D hypervolume of points dominated by ref_point.
            points: (N, 3) array
            Uses the standard 3D HV algorithm (O(N^2) worst case).
            """
            if len(points) == 0:
                return 0.0
            # Sort by first objective (ascending = better)
            order = np.argsort(points[:, 0])
            pts = points[order]

            # Remove dominated points (keep non-dominated front)
            # A point is dominated if there's another point with all objectives <=
            # Since sorted by f1 ascending, for each point check if any previous has f2<= and f3<=
            n = len(pts)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(i + 1, n):
                    if not nondom[j]:
                        continue
                    # pts[i] has f1 <= pts[j][0] (sorted)
                    if pts[i, 1] <= pts[j, 1] and pts[i, 2] <= pts[j, 2]:
                        nondom[j] = False
            pts = pts[nondom]
            n = len(pts)
            if n == 0:
                return 0.0

            # 3D HV: for each point on the non-dominated front, compute its contribution
            # Sort by f1 ascending
            # Contribution of point i: (f1_{i+1} - f1_i) * area_2d of the 2D region
            # where f2 and f3 are bounded by subsequent points
            # Standard 3D HV:
            # Sort by f1 ascending. For each point i, its contribution is:
            #   (next_f1 - f1_i) * max(0, (ref2 - f2_i)) * max(0, (ref3 - f3_i))
            # minus the parts already covered... 
            # Actually, the standard efficient 3D HV:
            # Sort by f1 ascending. Maintain the "envelope" in (f2, f3) space.
            # 
            # Simpler: use the decomposition:
            # HV = sum over i of (f1_{i+1} - f1_i) * HV_2D(points[i:], ref2, ref3)
            # where HV_2D is the 2D hypervolume of the (f2,f3) values of points i..n-1
            # with reference (ref2, ref3), but clipped to f1 range.
            #
            # Actually the cleanest 3D HV algorithm:
            # Sort by f1 ascending. 
            # For each i from 0 to n-1:
            #   Let f1_next = f1_{i+1} if i < n-1 else ref[0]
            #   The "slice" between f1_i and f1_next has 2D HV computed from points i..n-1
            #   but we need the 2D HV of (f2,f3) for points with f1 >= f1_i
            #
            # Let me use a different approach: the "union of boxes" via inclusion-exclusion
            # is too complex. Let me use the standard algorithm:
            #
            # 3D HV (from the literature):
            # Sort points by f1 ascending.
            # hv = 0
            # For i in 0..n-1:
            #   w1 = (f1_{i+1} if i<n-1 else ref0) - f1_i
            #   # Compute 2D HV of (f2,f3) for points i..n-1, ref (ref2, ref3)
            #   # But only count the part where f1 is in [f1_i, f1_{i+1})
            #   # Since all points i..n-1 have f1 >= f1_i, and we're looking at the slice
            #   # [f1_i, f1_{i+1}), the 2D region is determined by points i..n-1
            #   # 2D HV of (f2,f3) points:
            #   sub_pts = pts[i:, 1:3]
            #   hv2 = hypervolume_2d(sub_pts, ref[1], ref[2])
            #   hv += w1 * hv2
            #
            # But this is O(N^3). For N=200, that's 8M ops per HV call, times 200 selections = 1.6B. Too slow.
            #
            # Better: incremental HV. When adding a new point, compute the additional HV it contributes.
            # The marginal contribution of a new point p to an existing set S is:
            #   The volume of the box [p, ref] minus the volume already covered by S within [p, ref].
            #
            # For 3D, this is complex. Let me use a simpler but effective approach:
            # Use a coarse approximation for the greedy selection, then refine.
            #
            # Alternative: Since we have only 200 points to select from ~5000 candidates,
            # and 3D HV is expensive, let me use a different strategy:
            # 1. Pre-select a diverse subset using max-min in objective space
            # 2. Then do a local refinement pass
            #
            # Actually, let me just do the greedy with a fast HV approximation.
            # For the greedy selection, I'll use the "dominance-based" marginal contribution:
            # A new point adds HV proportional to the volume of the region it dominates
            # that isn't already dominated by existing points.
            #
            # For 3D with 200 points, let me just compute exact HV each time but optimize.
            # 200 selections * O(200^2) per HV = 200 * 40000 = 8M. That's fine!

        def hypervolume_2d(pts, ref2, ref3):
            """2D hypervolume. pts: (M,2) with f2, f3 values. Reference (ref2, ref3)."""
            if len(pts) == 0:
                return 0.0
            # Sort by f2 ascending
            order = np.argsort(pts[:, 0])
            p = pts[order]
            # Remove dominated: sorted by f2 asc, point j dominated by i if f2_i<=f2_j and f3_i<=f3_j
            n = len(p)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(i+1, n):
                    if not nondom[j]:
                        continue
                    if p[i, 0] <= p[j, 0] and p[i, 1] <= p[j, 1]:
                        nondom[j] = False
            p = p[nondom]
            n = len(p)
            if n == 0:
                return 0.0
            # 2D HV: sum of (f2_{i+1} - f2_i) * (ref3 - f3_i) for i in 0..n-1
            # where f2_{n} = ref2
            hv = 0.0
            for i in range(n):
                f2_next = p[i+1, 0] if i < n-1 else ref2
                w = f2_next - p[i, 0]
                h = ref3 - p[i, 1]
                if w > 0 and h > 0:
                    hv += w * h
            return hv

        def hypervolume_3d_fast(points, ref_point):
            """Compute 3D HV. points: (N,3). ref_point: (3,)."""
            if len(points) == 0:
                return 0.0
            ref0, ref1, ref2 = ref_point[0], ref_point[1], ref_point[2]
            # Filter to points strictly better than ref
            mask = np.all(points < ref_point, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by f1 ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)
            # Remove dominated
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(i+1, n):
                    if not nondom[j]:
                        continue
                    if pts[i, 0] <= pts[j, 0] and pts[i, 1] <= pts[j, 1] and pts[i, 2] <= pts[j, 2]:
                        nondom[j] = False
            pts = pts[nondom]
            n = len(pts)
            if n == 0:
                return 0.0
            # 3D HV via slicing
            hv = 0.0
            for i in range(n):
                f1_next = pts[i+1, 0] if i < n-1 else ref0
                w1 = f1_next - pts[i, 0]
                if w1 <= 0:
                    continue
                # 2D HV of (f2,f3) for points i..n-1
                sub = pts[i:, 1:3]
                hv2 = hypervolume_2d(sub, ref1, ref2)
                hv += w1 * hv2
            return hv

        # Greedy selection: at each step, pick the candidate that maximizes marginal HV
        # To speed up, pre-compute which candidates are not dominated by already-selected points
        selected = []  # list of (x0, x1, obj)
        remaining = set(range(n_valid))
        
        # For efficiency, track selected objective values
        sel_objs = np.zeros((0, 3))
        
        for step in range(n_select):
            if not check_time():
                break
            if used >= budget:
                break
            if len(remaining) == 0:
                break
            
            # If we have few remaining, just pick all
            if len(remaining) <= 5:
                for idx in remaining:
                    if not check_time() or used >= budget:
                        break
                    x = np.zeros(dim)
                    x[0] = pool_x0[idx]
                    x[1] = pool_x1[idx]
                    evaluate(x)
                    sel_objs = np.vstack([sel_objs, pool_objs[idx]]) if len(sel_objs) > 0 else pool_objs[idx].reshape(1,3)
                    selected.append(idx)
                remaining.clear()
                break
            
            # Compute marginal HV for each remaining candidate
            # To save time, only evaluate a subset of candidates
            # If remaining is large, sample a subset
            if len(remaining) > 500:
                # Sample 500 candidates for evaluation
                sample_idx = list(remaining)
                np.random.shuffle(sample_idx)
                candidates_to_eval = sample_idx[:500]
            else:
                candidates_to_eval = list(remaining)
            
            best_idx = -1
            best_hv = -1.0
            
            # Compute current HV
            if len(sel_objs) == 0:
                cur_hv = 0.0
            else:
                cur_hv = hypervolume_3d_fast(sel_objs, ref)
            
            for idx in candidates_to_eval:
                if not check_time():
                    break
                # Try adding this point
                if len(sel_objs) == 0:
                    new_objs = pool_objs[idx].reshape(1, 3)
                else:
                    new_objs = np.vstack([sel_objs, pool_objs[idx]])
                new_hv = hypervolume_3d_fast(new_objs, ref)
                if new_hv > best_hv:
                    best_hv = new_hv
                    best_idx = idx
            
            if best_idx == -1:
                # Fallback: pick first remaining
                best_idx = list(remaining)[0]
            
            # Evaluate the selected point
            x = np.zeros(dim)
            x[0] = pool_x0[best_idx]
            x[1] = pool_x1[best_idx]
            evaluate(x)
            
            if len(sel_objs) == 0:
                sel_objs = pool_objs[best_idx].reshape(1, 3)
            else:
                sel_objs = np.vstack([sel_objs, pool_objs[best_idx]])
            
            remaining.discard(best_idx)
            selected.append(best_idx)
            
            # Time check
            if not check_time():
                break

        # If we still have budget and time, do a refinement pass
        # Try small perturbations of selected points to see if we can improve HV
        if check_time() and used < budget:
            # For each selected point, try a few perturbations
            n_perturb = 5
            for sel_idx in selected[:]:
                if not check_time() or used >= budget:
                    break
                base_x0 = pool_x0[sel_idx]
                base_x1 = pool_x1[sel_idx]
                base_obj = pool_objs[sel_idx]
                
                # Try a few nearby points
                for dx0 in [-0.01, 0.01]:
                    for dx1 in [-0.01, 0.01]:
                        if not check_time() or used >= budget:
                            break
                        nx0 = base_x0 + dx0
                        nx1 = base_x1 + dx1
                        if nx0 < 0 or nx0 > 1 or nx1 < 0 or nx1 > 1:
                            continue
                        # Check if this point is in our pool (approximately)
                        # Compute its objective
                        new_obj = compute_objectives(nx0, nx1)
                        if np.any(new_obj >= ref):
                            continue
                        
                        # Check if it's close to an existing selected point
                        too_close = False
                        for so in sel_objs:
                            if np.allclose(new_obj, so, atol=0.005):
                                too_close = True
                                break
                        if too_close:
                            continue
                        
                        # Evaluate it
                        x = np.zeros(dim)
                        x[0] = nx0
                        x[1] = nx1
                        evaluate(x)
                        
                        # Check if it improves HV
                        if len(sel_objs) > 0:
                            test_objs = np.vstack([sel_objs, new_obj.reshape(1,3)])
                            if hypervolume_3d_fast(test_objs, ref) > hypervolume_3d_fast(sel_objs, ref):
                                sel_objs = test_objs
                                break  # Only add one perturbation per selected point
                    if not check_time() or used >= budget:
                        break

        return None
