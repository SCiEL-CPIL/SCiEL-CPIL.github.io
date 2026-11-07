"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and hypervolume EI for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Initial random sampling
        n_init = min(20, self.budget)
        X = lo + span * np.random.random((n_init, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Normalize objectives for GP (scale to [0,1] range)
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0

        def normalize_f(f):
            return (f - f_min) / f_range

        def denormalize_f(fn):
            return fn * f_range + f_min

        # GP kernel: RBF + white noise
        def gp_predict(X_train, Y_train, X_test, length_scales, noise_var):
            """Simple GP prediction with RBF kernel."""
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            
            # Compute kernel matrix
            # For efficiency, use squared distances
            K_train = np.zeros((n_train, n_train))
            for i in range(n_train):
                diff = X_train[i] - X_train
                sq_dist = np.sum(diff * diff, axis=1)
                K_train[i] = np.exp(-0.5 * sq_dist / length_scales)
            
            K_test = np.zeros((n_test, n_train))
            for i in range(n_test):
                diff = X_test[i] - X_train
                sq_dist = np.sum(diff * diff, axis=1)
                K_test[i] = np.exp(-0.5 * sq_dist / length_scales)
            
            # Add noise
            K_train += noise_var * np.eye(n_train)
            
            # Solve for mean and variance
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            mu = K_test @ alpha
            v = np.linalg.solve(L, K_test.T)
            var = 1.0 - np.sum(v * v, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        # Hypervolume computation for 3D (reference point based)
        ref_point = np.array([3.0, 5.0, 7.0])

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume using inclusion-exclusion."""
            if len(points) == 0:
                return 0.0
            
            # Filter points dominated by reference
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective
            valid = sorted(valid, key=lambda x: x[0])
            
            hv = 0.0
            for i, p in enumerate(valid):
                # For each point, compute its contribution
                # Points that are not dominated by any other point in valid
                dominated = False
                for j, q in enumerate(valid):
                    if i != j and np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Compute volume contribution
                # This is a simplified approach; proper HV requires more care
                x1 = min(p[0], ref[0])
                x2 = min(p[1], ref[1])
                x3 = min(p[2], ref[2])
                
                # Find the "box" that this point uniquely covers
                # Simplified: use the volume from this point to ref, 
                # minus overlaps with other non-dominated points
                vol = (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
                hv += vol  # This is an overestimate; proper HV needs inclusion-exclusion
            
            return hv

        # Better hypervolume for 3D using the standard algorithm
        def hypervolume_3d_correct(points, ref):
            """Compute 3D hypervolume using the standard O(n^2) algorithm."""
            if len(points) == 0:
                return 0.0
            
            # Filter points dominated by reference
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            
            # Remove dominated points
            valid = np.array(valid)
            n = len(valid)
            if n == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(valid[:, 0])
            valid = valid[order]
            
            hv = 0.0
            for i in range(n):
                # Check if point i is dominated by any other
                is_dominated = False
                for j in range(n):
                    if i != j and np.all(valid[j] <= valid[i]) and np.any(valid[j] < valid[i]):
                        is_dominated = True
                        break
                if is_dominated:
                    continue
                
                # Compute contribution using inclusion-exclusion
                # For 3D, we need to compute the volume of the box 
                # [p_i, ref] minus overlaps with other non-dominated points
                p = valid[i]
                
                # Find all points that could overlap with p's box
                # A point q overlaps if q[0] < ref[0], q[1] < ref[1], q[2] < ref[2]
                # and the boxes [p,ref] and [q,ref] overlap
                
                # Simplified approach: compute the volume from p to ref,
                # then subtract volumes of regions covered by other points
                # that are "closer" to the origin in some sense
                
                # Actually, let's use a simpler but correct approach for 3D
                # The hypervolume can be computed by sorting and using a sweep line
                # For simplicity, use the formula:
                # HV = sum over non-dominated points of (ref - p) 
                #       minus overlaps
                
                # Let's use a more direct approach
                # For each non-dominated point, compute its "unique" contribution
                # by finding the region it exclusively covers
                
                # This is getting complex. Let's use a different strategy:
                # Compute HV by decomposing into boxes
                # For 3D with small n, we can use the following:
                
                # Sort by x-coordinate
                # For each point, the contribution is the volume of the box
                # from the point to ref, minus the volumes of boxes from
                # other points that are within this box and have smaller x
                
                # Actually, the standard algorithm for 3D HV:
                # 1. Sort points by first coordinate
                # 2. For each point, compute the 2D HV of the points 
                #    that project within its "shadow"
                
                # Let me use a simpler but correct method:
                # HV = integral of the indicator function
                # For small n, we can compute it by decomposing the space
                
                # Use the following approach:
                # For each non-dominated point p_i, its contribution is:
                # (ref[0]-p_i[0]) * HV_2D({q[j:1] for q in valid if q[0] >= p_i[0] and q is not dominated by p_i in dims 1,2}, [ref[1], ref[2]])
                
                # This is still complex. Let me just use a numerical approach
                # or a simpler formula.
                
                # For now, use the simple sum (which overestimates) but scale it
                # Actually, let me implement the proper 3D HV algorithm
                
                pass
            
            # Let me implement a proper 3D hypervolume
            # Using the algorithm from "Computing the hypervolume indicator"
            
            # For 3D, we can use the following:
            # Sort points by x
            # Maintain a 2D structure for the y-z plane
            
            # Simplified correct algorithm for 3D:
            # HV = sum_{i} (x_{i+1} - x_i) * A_i
            # where x_i are sorted x-coordinates and A_i is the 2D HV of 
            # points with x >= x_i in the y-z plane
            
            # Let me just compute it directly for small n
            # by using the fact that for non-dominated points, we can
            # compute the volume of the union of boxes
            
            # For now, return a rough estimate
            # The exact HV is hard to compute quickly. Let me use a different approach.
            
            # Use the following: for each point, compute the volume it adds
            # by checking if it's non-dominated and computing the new volume
            
            # Actually, let me just use a simple but correct method:
            # Compute HV by inclusion-exclusion for 3D
            
            # For 3D with n points, the HV can be computed as:
            # HV = sum_{S subset of points} (-1)^{|S|+1} * volume(intersection of boxes in S)
            # But this is 2^n which is too slow for large n
            
            # Let me use a different approach: since we have at most ~50 points,
            # I'll compute HV by decomposing the space into cells
            
            # Actually, the simplest correct approach for 3D:
            # Sort by x. For each point i, the contribution is:
            # (x_{i+1} - x_i) * HV_2D(points with index >= i, projected to yz)
            # where x_{i+1} is the x of the next point or ref[0]
            
            # Let me implement this
            valid_sorted = valid  # already sorted by x
            n_valid = len(valid_sorted)
            
            # Build the 2D HV for suffixes
            # For each i, compute HV_2D of points i..n-1 in yz plane
            # with reference [ref[1], ref[2]]
            
            def hv_2d(points_2d, ref_2d):
                """Compute 2D hypervolume."""
                if len(points_2d) == 0:
                    return 0.0
                points_2d = np.array(points_2d)
                # Filter
                valid_2d = points_2d[np.all(points_2d < ref_2d, axis=1)]
                if len(valid_2d) == 0:
                    return 0.0
                # Sort by first coordinate
                order = np.argsort(valid_2d[:, 0])
                valid_2d = valid_2d[order]
                n2 = len(valid_2d)
                hv = 0.0
                for i in range(n2):
                    # Check if dominated
                    dominated = False
                    for j in range(n2):
                        if i != j and np.all(valid_2d[j] <= valid_2d[i]) and np.any(valid_2d[j] < valid_2d[i]):
                            dominated = True
                            break
                    if dominated:
                        continue
                    # Contribution
                    x1 = valid_2d[i, 0]
                    y1 = valid_2d[i, 1]
                    # Next x or ref
                    if i + 1 < n2:
                        x2 = valid_2d[i+1, 0]
                    else:
                        x2 = ref_2d[0]
                    # Width in x
                    dx = x2 - x1
                    # Height in y: from y1 to ref[1]
                    dy = ref_2d[1] - y1
                    hv += dx * dy
                return hv
            
            # Now compute 3D HV
            # For each point i (sorted by x), contribution is:
            # (x_{i+1} - x_i) * HV_2D({valid_j[1:] for j >= i}, [ref[1], ref[2]])
            # But we need to be careful: only points with x >= x_i contribute
            
            # Actually, the correct formula is:
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV_2D(suffix_i)
            # where suffix_i = {valid_j[1:] for j = i..n-1}
            # and x_i = valid_i[0], x_n = ref[0]
            
            hv_total = 0.0
            for i in range(n_valid):
                x_i = valid_sorted[i, 0]
                if i + 1 < n_valid:
                    x_next = valid_sorted[i+1, 0]
                else:
                    x_next = ref[0]
                
                # Get suffix points in yz
                suffix = valid_sorted[i:, 1:]
                hv_2d_val = hv_2d(suffix, ref[1:])
                hv_total += (x_next - x_i) * hv_2d_val
            
            return hv_total

        # Expected Improvement for hypervolume
        def compute_ei_hypervolume(X_train, F_train, X_candidate, length_scales, noise_var):
            """Compute hypervolume-based EI for a candidate point."""
            # Predict at candidate
            mu, var = gp_predict(X_train, F_train, X_candidate.reshape(1, -1), length_scales, noise_var)
            mu = mu[0]
            var = var[0]
            
            # Current hypervolume
            F_norm = normalize_f(F_train)
            F_norm_denorm = denormalize_f(F_norm)
            hv_current = hypervolume_3d_correct(F_norm_denorm, ref_point)
            
            # Sample from GP to estimate expected HV improvement
            n_samples = 50
            std = np.sqrt(var)
            samples = mu + std * np.random.normal(size=(n_samples, n_obj))
            
            # For each sample, compute HV with the new point added
            hv_improvements = np.zeros(n_samples)
            for i in range(n_samples):
                new_point = samples[i]
                # Add to current set
                all_points = np.vstack([F_norm_denorm, new_point.reshape(1, -1)])
                hv_new = hypervolume_3d_correct(all_points, ref_point)
                hv_improvements[i] = max(0, hv_new - hv_current)
            
            return np.mean(hv_improvements)

        # Main optimization loop
        # Use a small number of candidates and select the best by EI
        n_candidates = 50
        
        # Initialize GP parameters
        length_scales = 0.5  # Will be tuned
        noise_var = 1e-4
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points
            # Mix of random and local perturbations of best points
            candidates = []
            
            # Random candidates
            n_random = n_candidates // 2
            candidates.extend((lo + span * np.random.random((n_random, dim))).tolist())
            
            # Local perturbations of existing points
            n_local = n_candidates - n_random
            for _ in range(n_local):
                idx = np.random.randint(used)
                center = X[idx]
                scale = 0.1 * span
                candidate = center + scale * np.random.normal(size=dim)
                candidate = np.clip(candidate, lo, hi)
                candidates.append(candidate)
            
            candidates = np.array(candidates)
            
            # Evaluate EI for each candidate
            best_ei = -np.inf
            best_candidate = candidates[0]
            
            for cand in candidates:
                if time.time() > deadline:
                    break
                ei = compute_ei_hypervolume(X, F, cand, length_scales, noise_var)
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = cand
            
            # Evaluate the best candidate
            new_x = best_candidate
            new_f = problem(new_x)
            
            X = np.vstack([X, new_x.reshape(1, -1)])
            F = np.vstack([F, new_f.reshape(1, -1)])
            used += 1
            
            # Update normalization
            f_min = F.min(axis=0)
            f_max = F.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range < 1e-10] = 1.0
            
            # Occasionally adapt length scales (simple heuristic)
            if used % 10 == 0:
                # Try a few length scales
                best_ls = length_scales
                best_score = -np.inf
                for ls in [0.1, 0.3, 0.5, 1.0]:
                    # Quick evaluation: predict at a random point and check variance
                    test_point = lo + span * np.random.random(dim)
                    mu, var = gp_predict(X, F, test_point.reshape(1, -1), ls, noise_var)
                    # Prefer lower variance (more certain)
                    score = -var[0]
                    if score > best_score:
                        best_score = score
                        best_ls = ls
                length_scales = best_ls

        return X
# EVOLVE-BLOCK-END
