"""Surrogate-assisted multi-objective optimizer using GP + EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """GP-based multi-objective optimizer using Expected Hypervolume Improvement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initial sampling using Latin Hypercube for better coverage
        n_init = min(20, self.budget)
        X = self._latin_hypercube(n_init, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = n_init

        # GP Hyperparameters
        # Normalize inputs to [0, 1] for better conditioning of the kernel
        norm_lo = np.zeros(self.dim)
        norm_hi = np.ones(self.dim)
        
        # Length scales in normalized space
        length_scales = np.ones(self.dim) * 1.0
        signal_var = 1.0
        noise = 1e-5

        # Pre-compute for GP
        X_norm = (X - lo) / (hi - lo)
        
        # Cache for GP
        K_inv = None
        K_inv_y = np.zeros((self.n_obj, n_init))
        
        def update_gp_cache(X_norm, F, length_scales, signal_var, noise):
            """Update the GP cache (inverse kernel matrix and K_inv_y)."""
            n = X_norm.shape[0]
            K = self._compute_kernel(X_norm, X_norm, length_scales, signal_var)
            K[np.diag_indices_from(K)] += noise
            try:
                L = np.linalg.cholesky(K)
                K_inv = np.linalg.inv(K)
                K_inv_y = np.zeros((self.n_obj, n))
                for j in range(self.n_obj):
                    K_inv_y[j] = K_inv @ F[:, j]
            except np.linalg.LinAlgError:
                # Fallback to solve if Cholesky fails
                K_inv_y = np.zeros((self.n_obj, n))
                for j in range(self.n_obj):
                    K_inv_y[j] = np.linalg.solve(K, F[:, j])
                K_inv = np.linalg.inv(K)
            return K_inv, K_inv_y

        K_inv, K_inv_y = update_gp_cache(X_norm, F, length_scales, signal_var, noise)

        def predict_gp(X_new_norm, K_inv, K_inv_y, length_scales, signal_var, noise):
            """GP prediction for multiple points. Returns mean (n_obj, n_new) and std (n_obj, n_new)."""
            K_s = self._compute_kernel(X_new_norm, X_norm, length_scales, signal_var)
            # K_s shape: (n_new, n_train)
            # mean_j = K_s @ K_inv_y[j]
            means = np.zeros((self.n_obj, X_new_norm.shape[0]))
            for j in range(self.n_obj):
                means[j] = K_s @ K_inv_y[j]
            
            # Variance
            # v = K_s.T @ K_inv  (n_train, n_new)
            v = K_inv @ K_s.T
            K_ss = self._compute_kernel(X_new_norm, X_new_norm, length_scales, signal_var)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 0)
            stds = np.sqrt(var)
            
            # Reshape means to (n_new, n_obj) for convenience
            means = means.T
            stds = stds.reshape(-1, 1) # Broadcast to all objectives? No, each obj has its own GP.
            # Wait, the current setup trains separate GPs for each objective.
            # So stds should be (n_new, n_obj).
            # In my predict_gp above, I computed means for each objective separately.
            # I need to compute stds for each objective separately too?
            # Actually, the kernel is the same for all objectives, so the covariance structure (v) is the same.
            # Thus, the variance is the same for all objectives at a given point.
            # So stds is (n_new,) and can be broadcast.
            return means, stds

        def compute_hv(points, ref_point):
            """Calculate hypervolume for a set of points in 3D.
            Points are assumed to be non-dominated and better than ref_point.
            """
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than ref
            valid = np.all(points < ref_point, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            points = points[np.argsort(points[:, 0])]
            
            hv = 0.0
            # For 3D, we can use a sweep line or simple inclusion-exclusion.
            # Given small N, a simple approach:
            # For each point, calculate the volume of the box it "owns".
            # This is tricky in 3D. Let's use a simpler approximation or a library if available?
            # No library for HV in stdlib. 
            # Let's implement a simple 3D HV calculator.
            
            # Method: Sort by f1. Keep track of best f2 and f3.
            # Actually, a robust way for small N is to iterate through all subsets? No, too slow.
            # Let's use the "union of boxes" approach with inclusion-exclusion or a sweep.
            # Sweep line in f1:
            # We project to 2D (f2, f3).
            # As we sweep f1, we add points. The contribution of a point is (f1_next - f1_current) * HV_2D(points_active, ref_2d).
            # But points become "active" when they are non-dominated by previous points?
            # Actually, standard algorithm:
            # 1. Sort points by f1.
            # 2. Iterate through points. Maintain a list of "relevant" points (non-dominated in f2, f3).
            # 3. For each point i, the width in f1 is (points[i+1][0] - points[i][0]) if i < n-1 else (ref[0] - points[i][0]).
            #    Wait, the box extends to ref.
            #    The contribution is width * HV_2D(relevant_points, ref_2d).
            #    But "relevant_points" changes.
            
            # Simpler 3D HV for N < 100:
            # Use the fact that HV is the sum of volumes of disjoint boxes.
            # We can use a recursive partition or just a simple loop if N is small.
            # Given N_eval <= 200, N_front <= 200.
            # Let's try a simple O(N^3) or O(N^2 log N) method.
            
            # Sort by f1
            pts = points.copy()
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            # We maintain a list of "active" points in the 2D projection (f2, f3).
            # A point is active if it is not dominated by any other active point in f2 and f3.
            # When we move the f1 sweep line, we add points.
            # The hypervolume contribution of the slice between f1[i] and f1[i+1] is:
            # (f1[i+1] - f1[i]) * HV_2D(active_points, ref_2d)
            # However, the active set changes only when we add a point that is non-dominated.
            
            active = []
            for i in range(len(pts)):
                # Add point i to active set if it's non-dominated by existing active points
                p = pts[i]
                # Check if p is dominated by any in active
                dominated = False
                for a in active:
                    if a[1] <= p[1] and a[2] <= p[2] and (a[1] < p[1] or a[2] < p[2]):
                        dominated = True
                        break
                if not dominated:
                    # Remove points dominated by p
                    new_active = []
                    for a in active:
                        if not (p[1] <= a[1] and p[2] <= a[2] and (p[1] < a[1] or p[2] < a[2])):
                            new_active.append(a)
                    active = new_active
                    active.append(p)
                
                # Determine the next f1 value
                if i < len(pts) - 1:
                    f1_next = pts[i+1][0]
                else:
                    f1_next = ref_point[0]
                
                # Calculate HV_2D of active points
                hv_2d = self._hv_2d(active, ref_point[1:], )
                
                hv += (f1_next - p[0]) * hv_2d
                
            return hv

        def _hv_2d(points_2d, ref_2d):
            """Calculate 2D hypervolume for a list of non-dominated points."""
            if not points_2d:
                return 0.0
            # Sort by f2
            pts = sorted(points_2d, key=lambda x: x[1])
            hv = 0.0
            best_f3 = ref_2d[1]
            for i in range(len(pts)):
                p = pts[i]
                if p[2] < best_f3:
                    width = ref_2d[0] - p[1]
                    height = best_f3 - p[2]
                    hv += width * height
                    best_f3 = p[2]
            return hv

        def ehvi_candidate(x_new_norm, K_inv, K_inv_y, X_norm, F, length_scales, signal_var, noise, ref_point):
            """Calculate EHVI for a single candidate point."""
            x_new_norm = x_new_norm.reshape(1, -1)
            means, stds = predict_gp(x_new_norm, K_inv, K_inv_y, length_scales, signal_var, noise)
            # means shape (1, n_obj), stds shape (1, n_obj)
            mu = means[0]
            sd = stds[0]
            
            # Sample from predictive distribution
            n_samples = 20
            rng = np.random.RandomState(42)
            samples = mu + sd * rng.standard_normal((n_samples, self.n_obj))
            
            hv_improvements = []
            # Current HV
            # We need the current Pareto front to calculate current HV efficiently?
            # Or just calculate HV of F and HV of F U {sample}.
            # To save time, we can pre-calculate current HV once per iteration, not per candidate.
            # But here we are per candidate.
            # Let's assume we pass current_hv.
            
            for s in samples:
                if np.any(s >= ref_point):
                    continue
                # Check if s is dominated by any point in F
                # If dominated, HV improvement is 0
                # Optimization: Check against Pareto front of F
                # We don't have the front here easily without extra work.
                # Let's just check against all F. N=200, N_samples=20, N_cand=30. 200*20*30 = 120k ops. Acceptable.
                dominated = False
                for f in F:
                    if np.all(f <= s) and np.any(f < s):
                        dominated = True
                        break
                if dominated:
                    hv_improvements.append(0.0)
                    continue
                
                # Calculate new HV
                # Create new set of points
                # To speed up, we can just add s to F and calculate HV.
                # But HV calculation is O(N^2) or O(N^3).
                # Let's use a faster approximation:
                # The improvement is the volume of the box defined by s and ref, minus the volume already covered.
                # This is hard.
                # Let's stick to the full HV calculation but optimize it.
                # Since N is small, let's just do it.
                new_F = np.vstack([F, s])
                new_hv = compute_hv(new_F, ref_point)
                # We need current_hv to subtract.
                # We will pass current_hv as a global or argument.
                # For now, let's assume we calculate it outside and pass it in.
                # I will modify the signature.
                
            # This function is getting too complex for a single edit.
            # Let's restructure the main loop to be more efficient.
            
            return 0.0 # Placeholder

        # Main Loop
        # Pre-calculate current HV
        current_hv = compute_hv(F, ref_point)
        
        while used < self.budget:
            # Generate candidates
            n_candidates = 30
            X_cand = []
            
            # 1. Random candidates
            n_rand = 10
            X_cand_norm = np.random.random((n_rand, self.dim))
            X_cand.append(X_cand_norm)
            
            # 2. Candidates near high-variance regions
            # Sample points and evaluate their predictive variance
            n_var_samples = 50
            X_var_norm = np.random.random((n_var_samples, self.dim))
            # Get variance for these points
            _, stds_var = predict_gp(X_var_norm, K_inv, K_inv_y, length_scales, signal_var, noise)
            # stds_var shape (n_var_samples, n_obj). Take max std across objectives
            max_std = np.max(stds_var, axis=1)
            # Pick top 10
            top_var_idx = np.argsort(max_std)[-10:]
            X_cand.append(X_var_norm[top_var_idx])
            
            # 3. Candidates near best points (exploitation)
            # Find non-dominated points
            # Simple check:
            nd_idx = []
            for i in range(len(F)):
                is_nd = True
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            
            if len(nd_idx) > 0:
                # Pick a few non-dominated points to perturb
                n_perturb = min(5, len(nd_idx))
                pert_idx = np.random.choice(nd_idx, size=n_perturb, replace=False)
                for idx in pert_idx:
                    # Perturb in normalized space
                    perturb = X_norm[idx][None, :] + 0.1 * np.random.standard_normal((2, self.dim))
                    perturb = np.clip(perturb, 0, 1)
                    X_cand.append(perturb)
            
            X_cand = np.vstack(X_cand)
            X_cand = np.clip(X_cand, 0, 1)
            
            # Evaluate EHVI for each candidate
            # To make this fast, we need a fast EHVI.
            # Let's use a simplified EHVI:
            # EHVI(x) = E[HV(F U {f(x)}) - HV(F)]
            # We approximate E by sampling.
            # We need current_hv.
            
            ehvi_vals = []
            for x_c in X_cand:
                x_c = x_c.reshape(1, -1)
                means, stds = predict_gp(x_c, K_inv, K_inv_y, length_scales, signal_var, noise)
                mu = means[0]
                sd = stds[0]
                
                n_samples = 10
                rng = np.random.RandomState(42)
                samples = mu + sd * rng.standard_normal((n_samples, self.n_obj))
                
                hv_imps = []
                for s in samples:
                    if np.any(s >= ref_point):
                        hv_imps.append(0.0)
                        continue
                    # Check dominance
                    dominated = False
                    for f in F:
                        if np.all(f <= s) and np.any(f < s):
                            dominated = True
                            break
                    if dominated:
                        hv_imps.append(0.0)
                        continue
                    # Calculate new HV
                    # Optimization: Only calculate HV if s is non-dominated.
                    # We can speed up HV calculation by only considering the Pareto front of F.
                    # Let's maintain a list of non-dominated points in F.
                    # For now, use full F.
                    new_F = np.vstack([F, s])
                    new_hv = compute_hv(new_F, ref_point)
                    hv_imps.append(new_hv - current_hv)
                
                ehvi_vals.append(np.mean(hv_imps))
            
            ehvi_vals = np.array(ehvi_vals)
            best_idx = np.argmax(ehvi_vals)
            x_new_norm = X_cand[best_idx]
            x_new = lo + x_new_norm * (hi - lo)
            
            f_new = problem(x_new)
            used += 1
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            X_norm = np.vstack([X_norm, x_new_norm])
            
            # Update GP cache
            K_inv, K_inv_y = update_gp_cache(X_norm, F, length_scales, signal_var, noise)
            
            # Update current HV
            current_hv = compute_hv(F, ref_point)
            
            # Update hyperparameters
            if used % 20 == 0:
                # Simple adaptation
                length_scales = length_scales * 1.05
                length_scales = np.clip(length_scales, 0.1, 2.0)
                
        return X

    def _latin_hypercube(self, n, dim, lo, hi):
        """Generate n points in [lo, hi] using Latin Hypercube Sampling."""
        # For each dimension, divide [0, 1] into n bins and pick one random point from each bin
        X = np.zeros((n, dim))
        for j in range(dim):
            # Create bins
            bins = np.linspace(0, 1, n + 1)
            # Pick one random point from each bin
            for i in range(n):
                X[i, j] = bins[i] + np.random.random() * (bins[i+1] - bins[i])
        # Scale to [lo, hi]
        X = lo + X * (hi - lo)
        return X

    def _compute_kernel(self, X1, X2, length_scales, signal_var):
        """Compute GP kernel matrix with RBF kernel on normalized inputs."""
        # X1, X2 are normalized to [0, 1]
        X1s = X1 / length_scales
        X2s = X2 / length_scales
        sq_dists = np.sum(X1s ** 2, axis=1)[:, None] + np.sum(X2s ** 2, axis=1)[None, :] - 2 * X1s @ X2s.T
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var ** 2 * np.exp(-0.5 * sq_dists)
        return K
# EVOLVE-BLOCK-END
