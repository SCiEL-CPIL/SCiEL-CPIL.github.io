import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Generate dense candidate lattice (LHS)
        # 2000 points is a good balance for 4D
        n_candidates = 2000
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling
        # We can use scipy.stats.qmc.LatinHypercube
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        lhs_samples = sampler.random(n=n_candidates)
        candidates = lo + lhs_samples * (hi - lo)
        
        archive = []
        archive_objs = []
        evaluated_indices = set()
        
        X_train = []
        F_train = []
        
        # Helper: Fast 3D Hypervolume Calculation
        # Uses a sweep-line approach
        def hypervolume_3d(points, ref):
            if not points:
                return 0.0
            
            pts = np.array(points)
            # Filter points strictly better than reference
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            
            # We sweep along the first dimension.
            # The volume is the sum of slabs.
            # Each slab is defined by the f0 range where the set of "active" points (in f1-f2) is constant.
            # Actually, a simpler way for 3D:
            # Sort by f0.
            # Maintain a list of "active" points in the f1-f2 plane.
            # However, the standard sweep line for 3D HV is:
            # 1. Sort points by f0.
            # 2. Iterate through points.
            # 3. The contribution of a point is the volume of the box it dominates that is not already covered.
            # This is hard to do incrementally without a data structure.
            
            # Alternative: Recursive / Divide and Conquer is complex.
            # Let's use the "Sum of Boxes" approximation? No, must be exact.
            
            # Let's use a known O(N log N) or O(N^2) algorithm.
            # For N < 100, O(N^3) is fine.
            # But we call this many times.
            
            # Efficient 3D HV:
            # 1. Sort by f0.
            # 2. Use a 2D HV calculator for the "shadow" of the remaining points.
            # Vol = sum_{i} (f0[i] - f0[i-1]) * Area_2D(points[i:])
            # Where points[i:] are the points with f0 >= f0[i].
            # Wait, the "shadow" of points with f0 >= f0[i] in the f1-f2 plane is the 2D HV of those points.
            # The slab is from f0[i-1] to f0[i].
            # The cross section of the dominated region at f0 is the 2D region dominated by all points with f0_p <= f0.
            # So if we sort f0 ascending: p0, p1, ..., pN-1.
            # Slab 0: [p0[0], p1[0]]. Active points: {p0}. Area = HV_2D({p0[1:3]})
            # Slab 1: [p1[0], p2[0]]. Active points: {p0, p1}. Area = HV_2D({p0[1:3], p1[1:3]})
            # ...
            # Slab N-1: [pN-1[0], ref[0]]. Active points: {p0...pN-1}. Area = HV_2D(all)
            
            def hv_2d(points_2d, ref_2d):
                if len(points_2d) == 0:
                    return 0.0
                # points_2d: (N, 2)
                # Sort by x (first dim)
                idx = np.argsort(points_2d[:, 0])
                pts = points_2d[idx]
                
                area = 0.0
                max_y = ref_2d[1]
                # Sweep from right to left (highest x to lowest x)
                # Or left to right?
                # If we sort x ascending: x0 < x1 < ...
                # The region dominated by x_i is [x_i, ref_x] x [y_i, ref_y].
                # Overlaps occur.
                # Standard 2D HV:
                # Sort by x ascending.
                # Keep track of the minimum y seen so far? No.
                # Let's sweep x from high to low.
                # Sort x descending.
                idx_desc = np.argsort(points_2d[:, 0])[::-1]
                pts_desc = points_2d[idx_desc]
                
                min_y = ref_2d[1]
                area = 0.0
                for p in pts_desc:
                    if p[1] < min_y:
                        width = ref_2d[0] - p[0]
                        height = min_y - p[1]
                        area += width * height
                        min_y = p[1]
                    # If p[1] >= min_y, it is dominated in the projection by a point with higher x (processed earlier in desc sort) and lower y.
                
                return area

            total_vol = 0.0
            prev_f0 = pts[0][0]
            # We need to group points by f0? No, just iterate.
            # The "Active Set" for the slab [f0[i], f0[i+1]] is {p0, ..., p_i}.
            # So we iterate i from 0 to N-1.
            # Slab width: f0[i+1] - f0[i] (or ref[0] - f0[i] for last one)
            # Area: HV_2D of {p0[1:3], ..., p_i[1:3]}
            
            # To optimize, we can incrementally update the 2D HV?
            # 2D HV is not easily incremental.
            # But N is small (archive size ~ 20-50).
            # So O(N^2) for the 3D HV (calling hv_2d N times, each O(N log N)) is O(N^3).
            # 50^3 = 125,000.
            # We call this for ~200 candidates per step.
            # 200 * 125,000 = 25,000,000 ops per step.
            # 180 steps = 4.5 billion ops. Too slow.
            
            # We MUST optimize the candidate selection.
            # We will only calculate HV for a small subset of candidates (e.g., 50).
            
            return 0.0 # Placeholder, will be replaced by the loop logic

        # Helper: KNN Prediction
        def predict_knn(x_new, X_train, F_train, k=10):
            if len(X_train) == 0:
                return np.zeros(self.n_obj)
            
            dists = np.linalg.norm(X_train - x_new, axis=1)
            k = min(k, len(X_train))
            nn_indices = np.argpartition(dists, k)[:k]
            # Inverse distance weighting
            weights = 1.0 / (dists[nn_indices] + 1e-8)
            weights /= np.sum(weights)
            pred = np.dot(weights, F_train[nn_indices])
            return pred

        # Helper: Update Archive
        def update_archive(new_point, new_obj):
            # Check if dominated
            dominated = False
            for af in archive_objs:
                if np.all(af <= new_obj) and np.any(af < new_obj):
                    dominated = True
                    break
            
            if dominated:
                return archive, archive_objs
            
            # Remove dominated points
            new_archive = []
            new_archive_objs = []
            for p, f in zip(archive, archive_objs):
                if not (np.all(new_obj <= f) and np.any(new_obj < f)):
                    new_archive.append(p)
                    new_archive_objs.append(f)
            
            new_archive.append(new_point)
            new_archive_objs.append(new_obj)
            return new_archive, new_archive_objs

        # Initialization
        init_count = 20
        for i in range(init_count):
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            X_train.append(x)
            F_train.append(f)
            archive, archive_objs = update_archive(x, f)
        
        X_train = np.array(X_train)
        F_train = np.array(F_train)
        
        # Main Loop
        # To speed up, we select a subset of candidates for the greedy step
        subset_size = 100
        
        for step in range(init_count, self.budget):
            remaining = [i for i in range(n_candidates) if i not in evaluated_indices]
            if not remaining:
                break
            
            # Select a random subset of remaining candidates
            if len(remaining) <= subset_size:
                subset = remaining
            else:
                subset = rng.choice(remaining, size=subset_size, replace=False)
            
            best_marginal = -1.0
            best_idx = -1
            
            # Current HV
            # We need a fast HV calc. Let's implement a simple one for small N.
            def fast_hv_3d(pts, ref):
                if not pts: return 0.0
                pts = np.array(pts)
                pts = pts[np.all(pts < ref, axis=1)]
                if len(pts) == 0: return 0.0
                
                # Sort by f0
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                
                total_vol = 0.0
                # We will calculate the 2D HV of the "suffix" of points.
                # But to be fast, we'll just use the O(N^3) method if N is small, 
                # but we need it to be very fast.
                
                # Let's use the sweep line properly.
                # Vol = sum_{i=0}^{N-1} (f0_{i+1} - f0_i) * Area_2D(points[i:])
                # Where f0_N = ref[0].
                # points[i:] are points with index >= i.
                
                # This is still O(N^3) if we recompute Area_2D from scratch.
                # We can precompute the 2D HV for all suffixes?
                # Suffix 0: HV_2D(points[0:])
                # Suffix 1: HV_2D(points[1:])
                # ...
                # This is O(N^2) calls to HV_2D, which is O(N log N). Total O(N^3).
                
                # Given the time constraint, let's use a simpler approximation or a very optimized loop.
                # Actually, for N < 50, even a naive loop in Python might be slow.
                
                # Let's try to vectorize the 2D HV calculation.
                # 2D HV:
                # Sort by x desc.
                # Keep min y.
                # Area = sum (ref_x - x_i) * (min_y - y_i) for y_i < min_y.
                
                def hv_2d_fast(pts_2d, ref_2d):
                    if len(pts_2d) == 0: return 0.0
                    # pts_2d: (M, 2)
                    idx = np.argsort(pts_2d[:, 0])[::-1] # Desc x
                    pts = pts_2d[idx]
                    
                    min_y = ref_2d[1]
                    area = 0.0
                    # Vectorized update is hard due to dependency.
                    # But M is small.
                    for p in pts:
                        if p[1] < min_y:
                            area += (ref_2d[0] - p[0]) * (min_y - p[1])
                            min_y = p[1]
                    return area

                # 3D HV
                # We need Area_2D of suffixes.
                # Let's compute them.
                areas = np.zeros(len(pts))
                # We can compute HV_2D for the whole set, then remove points?
                # No, HV is not additive in that way.
                
                # Let's just loop.
                for i in range(len(pts)):
                    # Suffix: pts[i:]
                    # We need f0_{i+1} - f0_i.
                    if i == len(pts) - 1:
                        width = ref[0] - pts[i][0]
                    else:
                        width = pts[i+1][0] - pts[i][0]
                    
                    # Get 2D points for suffix
                    pts_2d_suffix = pts[i:, 1:3]
                    area = hv_2d_fast(pts_2d_suffix, ref[1:3])
                    total_vol += width * area
                
                return total_vol

            current_hv = fast_hv_3d(archive_objs, ref_point)
            
            for idx in subset:
                x = candidates[idx]
                f_pred = predict_knn(x, X_train, F_train)
                
                # Quick check: if dominated, skip
                dominated = False
                for af in archive_objs:
                    if np.all(af <= f_pred) and np.any(af < f_pred):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Calculate new HV
                new_objs = archive_objs + [f_pred]
                new_hv = fast_hv_3d(new_objs, ref_point)
                marginal = new_hv - current_hv
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx = idx
            
            if best_idx == -1:
                # Fallback to random
                best_idx = rng.choice(remaining)
            
            # Evaluate best
            x = candidates[best_idx]
            f = problem(x)
            evaluated_indices.add(best_idx)
            
            X_train = np.vstack([X_train, x])
            F_train = np.vstack([F_train, f])
            
            archive, archive_objs = update_archive(x, f)
            
        return archive
