# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization for WFG7 with GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 260  # safety margin before 300s/330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume
        ref_point = np.array([3.0, 5.0, 7.0])

        # Initial random sampling
        init_size = min(30, budget)
        X = lo + (hi - lo) * rng.random((init_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = init_size

        # Normalize objectives to [0,1] for GP
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range < 1e-10] = 1e-10
        F_norm = (F - F_min) / F_range

        def compute_hypervolume(points, ref):
            """Compute hypervolume of non-dominated points vs reference."""
            if len(points) == 0:
                return 0.0
            # Filter points dominated by ref
            valid = np.all(points < ref, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            # For 3D, use recursive subdivision
            def hv_3d(pts, ref):
                if len(pts) == 0:
                    return 0.0
                # Sort by first objective
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                
                hv = 0.0
                prev = pts[0]
                for i in range(1, len(pts)):
                    # Volume of box between prev and pts[i] in first dim
                    width = pts[i, 0] - prev[0]
                    # Project to 2D (y, z)
                    y_max = min(prev[1], ref[1])
                    z_max = min(prev[2], ref[2])
                    if y_max > prev[1] and z_max > prev[2]:
                        # This is simplified; proper 3D HV needs more care
                        pass
                    prev = pts[i]
                
                # Use a simpler approach: decompose into boxes
                # For each point, compute the volume it uniquely contributes
                total_hv = 0.0
                for i in range(len(pts)):
                    # Box from pts[i] to ref, minus overlaps with better points
                    # This is complex; use a grid-based approximation for small n
                    pass
                
                # Use scipy-free approach: sort and integrate
                # For 3 objectives, use the "slice" method
                # Sort by x
                pts_sorted = pts[np.argsort(pts[:, 0])]
                
                hv = 0.0
                # Integrate over x
                for i in range(len(pts_sorted)):
                    x_lo = pts_sorted[i, 0]
                    x_hi = pts_sorted[i + 1, 0] if i + 1 < len(pts_sorted) else ref[0]
                    if x_hi <= x_lo:
                        continue
                    # At this x-slice, find the max y and z from points with x <= current
                    # Actually, for non-dominated front, we need the Pareto front
                    # Let's use a different approach: enumerate all non-dominated points
                    # and compute the volume using the "corner" method
                    
                # Simpler: use the fact that for non-dominated points sorted by x,
                # the y and z values are anti-correlated. Use trapezoidal rule on the front.
                
                # For 3D HV, use the algorithm from:
                # "Computing the hypervolume indicator"
                # We'll use a simple grid-based approximation
                n_grid = 20
                x_grid = np.linspace(pts[:, 0].min(), ref[0], n_grid)
                y_grid = np.linspace(pts[:, 1].min(), ref[1], n_grid)
                z_grid = np.linspace(pts[:, 2].min(), ref[2], n_grid)
                
                # This is too slow. Let's use a simpler exact method for small n.
                
                # Exact 3D HV using recursive decomposition
                def _hv(points, ref):
                    n = len(points)
                    if n == 0:
                        return 0.0
                    if n == 1:
                        return max(0, ref[0] - points[0, 0]) * max(0, ref[1] - points[0, 1]) * max(0, ref[2] - points[0, 2])
                    
                    # Find the point that minimizes the first objective
                    idx = np.argmin(points[:, 0])
                    p = points[idx]
                    others = np.delete(points, idx, axis=0)
                    
                    # Split: points dominated by p in first obj but not fully
                    # The volume = box(p, ref) + volume of points not dominated by p in the region beyond p
                    box_vol = max(0, ref[0] - p[0]) * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
                    
                    # Points that are not dominated by p: those with some objective < p
                    # This recursive approach is complex. Let's use a different method.
                    
                    return box_vol  # This is an under-estimate
                
                # Use a practical approach: for our small problem, use the fact that
                # WFG7 has a smooth Pareto front. Use a simple sampling-based HV estimate
                # But we need exact for scoring. Let's implement a proper 3D HV.
                
                # Proper 3D HV algorithm (from literature)
                def hv_3d_exact(pts, ref):
                    n = len(pts)
                    if n == 0:
                        return 0.0
                    if n == 1:
                        return float(np.prod(np.maximum(0, ref - pts[0])))
                    
                    # Sort by first objective
                    order = np.argsort(pts[:, 0])
                    pts = pts[order]
                    
                    hv = 0.0
                    # For each consecutive pair in sorted order, compute the contribution
                    # This is still complex. Let's use the "slicing" approach.
                    
                    # Slice by x-coordinate
                    x_vals = np.unique(np.concatenate([pts[:, 0], [ref[0]]]))
                    
                    for i in range(len(x_vals) - 1):
                        x_lo = x_vals[i]
                        x_hi = x_vals[i + 1]
                        dx = x_hi - x_lo
                        if dx <= 0:
                            continue
                        
                        # Find points with x <= x_hi (i.e., in this slice or before)
                        # The "active" points are those with x <= x_hi
                        active = pts[pts[:, 0] <= x_hi]
                        if len(active) == 0:
                            continue
                        
                        # In the 2D projection (y, z), compute the 2D HV of active points
                        # with reference (ref[1], ref[2])
                        # But we only count the part where x is in [x_lo, x_hi]
                        # This is getting complicated.
                        
                        # Let's use a different strategy: since we have at most ~200 points,
                        # use a Monte Carlo estimate with enough samples for accuracy.
                        pass
                    
                    # Monte Carlo HV estimate
                    n_samples = 100000
                    x_s = rng.uniform(pts[:, 0].min(), ref[0], n_samples)
                    y_s = rng.uniform(pts[:, 1].min(), ref[1], n_samples)
                    z_s = rng.uniform(pts[:, 2].min(), ref[2], n_samples)
                    
                    # A point (x, y, z) is in the HV if there exists a point p in pts
                    # such that p[0] <= x, p[1] <= y, p[2] <= z
                    # And x < ref[0], y < ref[1], z < ref[2]
                    
                    count = 0
                    for i in range(n_samples):
                        x, y, z = x_s[i], y_s[i], z_s[i]
                        # Check if dominated by any point
                        for j in range(len(pts)):
                            if pts[j, 0] <= x and pts[j, 1] <= y and pts[j, 2] <= z:
                                count += 1
                                break
                    
                    total_vol = (ref[0] - pts[:, 0].min()) * (ref[1] - pts[:, 1].min()) * (ref[2] - pts[:, 2].min())
                    return total_vol * count / n_samples
                
                return hv_3d_exact(points, ref)
            
            return hv_3d(points, ref)

        def get_pareto_front(F):
            """Return indices of non-dominated points."""
            n = len(F)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return np.where(nondom)[0]

        # GP-based acquisition
        def fit_gp(X_train, F_train):
            """Fit a simple GP using kernel methods."""
            from scipy.stats import norm
            n = len(X_train)
            d = X_train.shape[1]
            
            # RBF kernel
            def kernel(X1, X2, length_scale=0.5, sigma=1.0):
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                return sigma ** 2 * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            
            K = kernel(X_train, X_train, 0.5, 1.0)
            K[np.diag_indices_from(K)] += 1e-8
            
            # For multi-output, fit separate GPs for each objective
            means = []
            covs = []
            for obj in range(n_obj):
                y = F_train[:, obj]
                # Standardize
                y_mean = y.mean()
                y_std = y.std() + 1e-10
                y_std = max(y_std, 0.1)
                y_norm = (y - y_mean) / y_std
                
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    means.append((y_mean, y_std, L, alpha))
                except np.linalg.LinAlgError:
                    # Fallback: use mean only
                    means.append((y.mean(), 1.0, None, None))
            
            return means, X_train

        def predict_gp(gp_params, X_new):
            """Predict mean and std at new points."""
            means_params, X_train = gp_params
            predictions = []
            for obj in range(n_obj):
                y_mean, y_std, L, alpha = means_params[obj]
                if L is None:
                    pred_mean = np.full(len(X_new), y_mean)
                    pred_std = np.full(len(X_new), y_std)
                else:
                    K_s = np.zeros((len(X_new), len(X_train)))
                    for i in range(len(X_new)):
                        diff = X_new[i] - X_train
                        sq_dist = np.sum(diff ** 2, axis=1)
                        K_s[i] = np.exp(-0.5 * sq_dist / (0.5 ** 2))
                    
                    v = np.linalg.solve(L, K_s.T)
                    pred_mean = (y_mean + y_std * (K_s @ alpha))
                    K_ss = np.ones((len(X_new), len(X_new)))
                    for i in range(len(X_new)):
                        for j in range(i + 1, len(X_new)):
                            diff = X_new[i] - X_new[j]
                            sq_dist = np.sum(diff ** 2)
                            K_ss[i, j] = K_ss[j, i] = np.exp(-0.5 * sq_dist / (0.5 ** 2))
                    pred_var = 1.0 - np.sum(v ** 2, axis=1)
                    pred_var = np.maximum(pred_var, 0.0)
                    pred_std = y_std * np.sqrt(pred_var)
                
                predictions.append((pred_mean, pred_std))
            return predictions

        # EHVI acquisition (simplified)
        def ehvi_acquisition(X_candidates, gp_params, F_current, ref_point):
            """Compute EHVI for candidate points."""
            n_cand = len(X_candidates)
            predictions = predict_gp(gp_params, X_candidates)
            
            # Current HV
            nondom_idx = get_pareto_front(F_current)
            F_pareto = F_current[nondom_idx]
            hv_current = compute_hypervolume(F_pareto, ref_point)
            
            ehvi_scores = np.zeros(n_cand)
            
            # Sample-based EHVI
            n_samples = 100
            for s in range(n_samples):
                # Sample from GP
                F_samples = np.zeros((n_cand, n_obj))
                for obj in range(n_obj):
                    pred_mean, pred_std = predictions[obj]
                    F_samples[:, obj] = pred_mean + pred_std * rng.standard_normal(n_cand)
                
                # For each candidate, compute HV if added to current set
                for i in range(n_cand):
                    F_new = np.vstack([F_current, F_samples[i:i + 1]])
                    nondom_new = get_pareto_front(F_new)
                    F_pareto_new = F_new[nondom_new]
                    hv_new = compute_hypervolume(F_pareto_new, ref_point)
                    improvement = hv_new - hv_current
                    ehvi_scores[i] += max(0, improvement)
            
            ehvi_scores /= n_samples
            return ehvi_scores

        # Main BO loop
        while used < budget:
            if time.time() > deadline:
                break

            # Fit GP
            gp_params = fit_gp(X, F)

            # Generate candidates
            n_candidates = 50
            X_cand = lo + (hi - lo) * rng.random((n_candidates, dim))
            
            # Also add local perturbations around current best points
            nondom_idx = get_pareto_front(F)
            if len(nondom_idx) > 0:
                n_local = min(20, n_candidates - 10)
                for i in range(n_local):
                    base_idx = nondom_idx[rng.integers(len(nondom_idx))]
                    perturbation = 0.1 * (hi - lo) * rng.standard_normal(dim)
                    X_cand[i] = np.clip(X[base_idx] + perturbation, lo, hi)
                n_candidates = 10 + n_local

            # Evaluate EHVI
            ehvi = ehvi_acquisition(X_cand, gp_params, F, ref_point)
            best_cand = X_cand[np.argmax(ehvi)]

            # Evaluate
            fc = problem(best_cand)
            used += 1

            # Add to archive
            X = np.vstack([X, best_cand])
            F = np.vstack([F, fc])

            # Update normalization
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range < 1e-10] = 1e-10

        return X
# EVOLVE-BLOCK-END
