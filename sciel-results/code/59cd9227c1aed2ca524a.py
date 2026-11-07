"""Multi-objective optimizer on WFG7: Bayesian Optimization with EHVI acquisition."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import NearestNDInterpolator
import sys

class Optimizer:
    """Bayesian Multi-objective Optimizer using GP surrogate and EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # 1. Initial Sampling
        # Latin Hypercube Sampling for better initial coverage
        n_init = 20
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(n_init)
            X_init[:, i] = (perm + 0.5) / n_init
        X_init = X_init * (hi - lo) + lo
        
        F_init = np.array([problem(x) for x in X_init])
        all_F = F_init.copy()
        used = n_init
        
        # 2. Bayesian Optimization Loop
        # We will select 10 points per iteration
        batch_size = 10
        max_iter = 100 # Safety cap, loop will stop by budget or time
        
        # GP Hyperparameters
        # Standard deviation of the noise (small)
        noise_var = 1e-5
        # Length scales: start with 0.5 (normalized space)
        # Signal variance: 1.0
        theta = np.array([0.5] * self.dim + [1.0, noise_var]) # [length_scales..., signal_var, noise_var]
        
        # Store data for GP
        X_data = X_init.copy()
        F_data = F_init.copy()
        
        # Helper: RBF Kernel
        def rbf_kernel(X1, X2, length_scales):
            # X1: (n, d), X2: (m, d)
            # Normalize inputs to [0, 1] for kernel stability
            X1_norm = (X1 - lo) / (hi - lo)
            X2_norm = (X2 - lo) / (hi - lo)
            
            # Compute squared distances
            # (n, 1, d) * (1, m, d) -> sum over d
            diff = X1_norm[:, np.newaxis, :] - X2_norm[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            # RBF: exp(-0.5 * sum((x-y)^2 / l^2))
            # l^2 is shape (d,)
            sq_dist_scaled = sq_dist / np.sum(length_scales**2, axis=0) # This is wrong, need per-dim scaling
            # Correct: sum_d (x_d - y_d)^2 / l_d^2
            # diff is (n, m, d). length_scales is (d,).
            # diff_sq_scaled = diff**2 / length_scales**2
            diff_sq_scaled = diff**2 / length_scales[np.newaxis, np.newaxis, :]**2
            sq_dist_scaled = np.sum(diff_sq_scaled, axis=2)
            
            return np.exp(-0.5 * sq_dist_scaled)

        def compute_cov(X1, X2, theta):
            ls = theta[:self.dim]
            signal_var = theta[-2]
            noise_var = theta[-1]
            K = signal_var * rbf_kernel(X1, X2, ls)
            if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
                K[np.diag_indices_from(K)] += noise_var
            return K

        def compute_cholesky(K):
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape[0]))
                return L, True
            except np.linalg.LinAlgError:
                # Add jitter
                for jitter in [1e-6, 1e-5, 1e-4, 1e-3]:
                    try:
                        L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
                        return L, True
                    except np.linalg.LinAlgError:
                        continue
                # Fallback: diagonal approximation or identity
                L = np.sqrt(np.diag(np.diag(K)))
                return L, False

        # Hypervolume Calculation (3D)
        # Input: points (n, 3), ref_point (3,)
        # Returns: HV
        def hypervolume_3d(points, ref):
            # Filter points dominated by ref
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            pts = points[valid]
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # 2D HV of remaining points projected on obj2, obj3
            # For each point i, the contribution is (ref[0] - pts[i][0]) * HV_2d(pts[i+1:], ref[1:])
            # This is O(N^2) which is fine for N<=200.
            n = len(pts)
            hv = 0.0
            for i in range(n):
                # The box starting at pts[i] and ending at ref
                # But we must exclude parts covered by points with smaller obj1 (already processed? No, sorted ascending)
                # Actually, standard sweep line:
                # Sort by x1.
                # For each point i, the volume it uniquely contributes is:
                # (ref_x1 - x1_i) * HV_2d( points j > i with x2_j > x2_i and x3_j > x3_i )? 
                # No, simpler:
                # HV = sum over i of (ref_x1 - x1_i) * (area of union of rectangles from points j>i in x2-x3 plane that are "above" point i?)
                # Let's use a robust O(N^2) or O(N^3) method for 3D.
                # Given N is small (< 200), O(N^3) is ~ 8M ops, acceptable in Python if optimized? Maybe slow.
                # Let's use the standard 3D HV algorithm:
                # 1. Sort by x1.
                # 2. Iterate i from 0 to N-1.
                # 3. For point i, consider all points j > i.
                # 4. The contribution of point i is (ref_x1 - x1_i) * (2D HV of points j>i restricted to x2 > x2_i and x3 > x3_i, relative to ref[1:])?
                # Actually, it's easier to just compute the union of boxes.
                # But for 3D, the "sweep" method:
                # Sort by x1.
                # Keep a list of active "fronts" in x2-x3.
                # This is complex.
                
                # Alternative: Monte Carlo? No, exact required.
                # Let's use a simple O(N^2) for 2D and O(N^3) for 3D?
                # 2D HV:
                def hv_2d(pts2, ref2):
                    # pts2: (n, 2), ref2: (2,)
                    pts2 = pts2[np.all(pts2 < ref2, axis=1)]
                    if len(pts2) == 0: return 0.0
                    pts2 = pts2[np.argsort(pts2[:, 0])]
                    hv = 0.0
                    prev_y = ref2[1]
                    for p in pts2:
                        if p[1] < prev_y:
                            hv += (ref2[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                        # If p[1] >= prev_y, it's dominated in y by a previous point?
                        # Wait, sorted by x asc. If y is also asc, the last point has largest x and largest y.
                        # The points that contribute are those on the Pareto front.
                        # If sorted by x asc, the Pareto front has y desc.
                        # So we iterate x asc. We keep track of the minimum y seen so far? No.
                        # The area is sum of rectangles.
                        # Point i contributes (ref_x - x_i) * (y_{i-1} - y_i) where y_{i-1} is the y of the previous point in the sorted-by-x list that is on the front.
                        # Actually, just filter non-dominated first.
                    return hv
                
                # Let's implement a robust 3D HV using the standard algorithm:
                # Sort by x1.
                # For each point i, the volume is (ref_x1 - x1_i) * (HV_2D of points j>i with x2_j > x2_i and x3_j > x3_i, ref=(ref_x2, ref_x3))
                # This is not quite right because points j>i might have smaller x2.
                
                # Correct Algorithm for 3D HV:
                # 1. Sort points by x1.
                # 2. Iterate i from 0 to N-1.
                # 3. Maintain a set of points j > i that are not dominated by any k > i.
                # 4. The contribution of point i is (ref_x1 - x1_i) * HV_2D( {j > i | x2_j > x2_i, x3_j > x3_i}, ref_2d )?
                # No. The volume "behind" point i in the x1 direction is filled by the "shadow" of the points that are better in x1 (i.e., j > i).
                # The cross-section at x1_i is the 2D HV of the points j>i projected onto x2-x3, BUT only the part that is "above" point i in x2 and x3?
                # Actually, the volume contributed by point i is the volume of the box [x1_i, ref_x1] x [x2_i, ref_x2] x [x3_i, ref_x3] MINUS the volume covered by points j>i.
                # This is getting complicated to implement correctly from scratch quickly.
                
                # Let's use a simpler, slightly slower but correct method:
                # Inclusion-Exclusion or just summing disjoint boxes.
                # Sort by x1.
                # For each point i, define the box B_i = [x1_i, ref_x1] x [x2_i, ref_x2] x [x3_i, ref_x3].
                # The total HV is the volume of the union of B_i.
                # Since N is small, we can use a grid? No, exact.
                
                # Let's use the "Sweep Line" for 3D.
                # Sort by x1.
                # We maintain a 2D structure of the "current front" in x2-x3.
                # When we move past x1_i, we add point i to the structure? No, we remove it?
                # Actually, the standard algorithm is:
                # Sort by x1.
                # Iterate i from N-1 down to 0.
                # Maintain a list of points that are non-dominated among {i+1, ..., N-1}.
                # The contribution of point i is (ref_x1 - x1_i) * (2D HV of the non-dominated points among {j>i} that are "dominated" by i in x2, x3? No.)
                
                # Let's use a library-like implementation if possible? No, stdlib only.
                # I will implement a O(N^2) HV for 3D based on:
                # HV = sum_{i} (ref_x1 - x1_i) * A_i
                # where A_i is the 2D area of the union of rectangles [x2_j, ref_x2]x[x3_j, ref_x3] for all j such that x1_j > x1_i AND x2_j > x2_i AND x3_j > x3_i?
                # No, that's not right.
                
                # Let's use the algorithm from "Computing the Hypervolume Indicator":
                # 1. Sort points by x1.
                # 2. For each point i, calculate the 2D HV of the points j > i, but only considering the region x2 > x2_i and x3 > x3_i?
                # Actually, the contribution of point i is the volume of the box defined by i and ref, minus the volume covered by points j>i.
                # The points j>i cover the region where x1 > x1_i.
                # The intersection of the box of i and the union of boxes of j>i is the region where x1 > x1_i, x2 > x2_i, x3 > x3_i.
                # This region is exactly the 2D HV of the points j>i projected on x2-x3, restricted to x2 > x2_i and x3 > x3_i?
                # No, the points j>i have their own x2, x3.
                # The volume covered by j>i in the slice x1 > x1_i is the 2D HV of {j>i} in the x2-x3 plane, with reference point (ref_x2, ref_x3).
                # BUT, we only care about the part of that 2D HV that lies in x2 > x2_i and x3 > x3_i?
                # No, if a point j has x2_j < x2_i, its box [x2_j, ref_x2] covers x2_i. So it contributes to the volume "above" i.
                # So, Contribution(i) = (ref_x1 - x1_i) * (2D HV of {j>i} in x2-x3 plane, ref=(ref_x2, ref_x3))
                # Wait, if j has x2_j < x2_i, then the box of j in x2-x3 is [x2_j, ref_x2]x[x3_j, ref_x3].
                # This box covers the region x2 > x2_i.
                # So yes, the entire 2D HV of the set {j>i} is the volume of the union of their boxes in x2-x3.
                # And since all these boxes extend to ref_x2, ref_x3, they all cover the region "above" any point with x2 < x2_i?
                # No. If x2_j < x2_i, the box starts at x2_j, so it covers x2_i.
                # If x2_j > x2_i, the box starts at x2_j, so it does NOT cover x2_i.
                # So the volume covered by {j>i} in the region x2 > x2_i, x3 > x3_i is NOT simply the 2D HV.
                # It is the 2D HV of {j>i} INTERSECTED with [x2_i, ref_x2] x [x3_i, ref_x3].
                
                # This is effectively the 2D HV of the points {j>i} with a modified reference point?
                # No.
                
                # Let's use a simpler O(N^3) method for N=200.
                # 200^3 = 8,000,000. In Python, this might take 1-2 seconds.
                # We call this function ~10 times per BO iteration (for acquisition optimization).
                # 10 * 2s = 20s per iteration. 10 iterations = 200s. Too slow.
                
                # We need a faster HV.
                # Let's use the "Sweep Line" for 2D, and for 3D, use the property:
                # HV_3D = sum_{i} (ref_x1 - x1_i) * HV_2D( {j>i} , ref_2d )
                # IS THIS TRUE?
                # Let's check.
                # The volume is the integral of the cross-sectional area A(x1) from 0 to ref_x1.
                # A(x1) is the 2D HV of the points with x1_j > x1.
                # Since the points are discrete, A(x1) is constant between x1_i and x1_{i-1}.
                # So HV = sum_{i} (x1_{i-1} - x1_i) * A(x1_i)
                # Where A(x1_i) is the 2D HV of points {j | x1_j > x1_i}.
                # Yes! This is correct.
                
                # So we need a fast 2D HV.
                # 2D HV is O(N log N) or O(N).
                # O(N) if sorted.
                
                def hv_2d_fast(pts, ref):
                    # pts: (n, 2), ref: (2,)
                    pts = pts[np.all(pts < ref, axis=1)]
                    if len(pts) == 0: return 0.0
                    pts = pts[np.argsort(pts[:, 0])]
                    hv = 0.0
                    # The Pareto front in 2D sorted by x asc has y desc.
                    # We can just iterate and keep the min y seen so far? No.
                    # We iterate x asc.
                    # The area is sum of (ref_x - x_i) * (y_{i-1} - y_i) for points on the front.
                    # Points on the front are those where y is strictly decreasing as x increases.
                    # Let's filter non-dominated.
                    # Since sorted by x, a point is non-dominated if its y is less than all previous y's.
                    # Wait, if x is smaller, it's "better". So if we sort x asc, the first point has smallest x.
                    # It dominates all points with larger x and larger y.
                    # So the non-dominated points are those where y is strictly decreasing.
                    front = []
                    min_y = np.inf
                    for p in pts:
                        if p[1] < min_y:
                            front.append(p)
                            min_y = p[1]
                    # Now calculate HV of front
                    # front is sorted by x asc, y desc.
                    # Add ref point
                    front.append([ref[0], ref[1]])
                    hv = 0.0
                    for i in range(len(front) - 1):
                        x1, y1 = front[i]
                        x2, y2 = front[i+1]
                        hv += (x2 - x1) * (ref[1] - y1)
                    return hv

                # Now implement HV_3D using this.
                # Sort by x1.
                # Iterate i from 0 to N-1.
                # A_i = hv_2d_fast( pts[i+1:, 1:], ref[1:] )
                # HV += (ref[0] - pts[i][0]) * A_i  -- WAIT.
                # The formula is: HV = sum_{i} (x1_{i-1} - x1_i) * A(x1_i)
                # Let's sort x1 desc? Or asc?
                # If sorted asc: x_0 < x_1 < ... < x_{N-1} < ref_x1.
                # A(x) is constant for x in [x_i, x_{i+1}).
                # The value is A_i = 2D HV of points {j | x1_j > x_i}.
                # So HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * A_i  + (ref_x1 - x_{N-1}) * A_{N-1}
                # Where x_{-1} = 0? No, the domain starts at 0? Or at the min x?
                # The reference point defines the upper bound. The lower bound is 0 (or min x).
                # Actually, the volume is from 0 to ref_x1.
                # If no points exist in [0, x_0), A(x) = 0? No, if points have x > x_0, they contribute.
                # So A(x) for x < x_0 is the 2D HV of ALL points.
                # So we should start from x=0.
                # Let's define x_0 = 0.
                # Then term 0: (x_1 - 0) * A(0) where A(0) is 2D HV of all points.
                # Term i: (x_{i+1} - x_i) * A(x_i) where A(x_i) is 2D HV of points with x1 > x_i.
                # Last term: (ref_x1 - x_{N-1}) * A(x_{N-1}).
                
                pts_sorted = pts[np.argsort(pts[:, 0])]
                n = len(pts_sorted)
                if n == 0: return 0.0
                
                hv = 0.0
                # Precompute 2D HVs?
                # A_i depends on suffix i+1.
                # We can compute A_i for i from N-1 down to 0.
                # A_{N-1} = 0 (no points after last)
                # A_{N-2} = 2D HV of {N-1}
                # ...
                # A_0 = 2D HV of {1..N-1}
                # And we need A_{-1} = 2D HV of {0..N-1} for the first segment [0, x_0].
                
                # Let's just compute A_i on the fly. O(N^2) for 3D HV total.
                # For each i, compute 2D HV of pts[i+1:].
                # 2D HV is O(N). So total O(N^2).
                # N=200 -> 40,000 ops. Very fast.
                
                x1s = pts_sorted[:, 0]
                x23 = pts_sorted[:, 1:]
                
                # Segment 1: [0, x1s[0]]
                # A is 2D HV of all points
                hv_seg = hv_2d_fast(x23, ref[1:])
                hv += x1s[0] * hv_seg
                
                # Segments i: [x1s[i], x1s[i+1]]
                for i in range(n - 1):
                    # A is 2D HV of points i+1 to end
                    hv_seg = hv_2d_fast(x23[i+1:], ref[1:])
                    hv += (x1s[i+1] - x1s[i]) * hv_seg
                    
                # Last segment: [x1s[-1], ref_x1]
                # A is 0 (no points after last)
                # hv += (ref[0] - x1s[-1]) * 0
                
                return hv

        # Acquisition: EHVI
        # Approximate EHVI using Monte Carlo or Gaussian quadrature?
        # With 3 objectives, we can use a grid.
        # Or, simpler: Use "Expected Improvement" on a scalarization?
        # No, EHVI is better.
        # Let's use a simple grid-based EHVI approximation.
        # Grid size: 10x10x10 = 1000 points.
        # For each candidate x, compute GP mean and cov.
        # Sample 50 points from the GP?
        # 1000 candidates * 50 samples * 3 obj = 150,000 ops per candidate.
        # 10 candidates per batch. 10 * 150,000 = 1.5M ops. Fast.
        
        # But we need to optimize the acquisition function over X.
        # We can use a grid search over X as well?
        # Grid size for X: 5^12 is too big.
        # Use random sampling for candidates?
        # Generate 200 random candidates, compute EHVI for each, pick top 10.
        
        # GP Prediction
        def gp_predict(X_new, X_data, F_data, theta):
            # X_new: (n, d)
            # X_data: (m, d)
            # F_data: (m, 3)
            
            K = compute_cov(X_data, X_data, theta)
            L, _ = compute_cholesky(K)
            
            K_s = compute_cov(X_data, X_new, theta) # (m, n)
            K_ss = compute_cov(X_new, X_new, theta) # (n, n)
            
            # Mean
            # alpha = L^-T (L^-1 K_s^T)
            # Solve L y = K_s^T
            K_s_T = K_s.T
            y = np.linalg.solve(L, K_s_T)
            alpha = np.linalg.solve(L.T, y)
            
            mean = alpha.T @ F_data # (n, 3)
            
            # Cov
            # v = L^-1 K_s^T
            v = np.linalg.solve(L, K_s_T)
            cov = K_ss - v.T @ v # (n, 3) x (n, 3)?
            # K_s is (m, n). K_s_T is (n, m).
            # v is (m, n).
            # v.T is (n, m).
            # v.T @ v is (n, n). This is the cross-covariance matrix for the same objective?
            # We need the covariance between objectives for the same point?
            # GP is usually univariate. Here we have 3 objectives.
            # Assume independent GPs for each objective?
            # Or a multi-output GP?
            # Let's assume independent GPs for simplicity.
            # Then Cov(f_i, f_j) = 0 for i != j.
            # Cov(f_i, f_i) = K_ss - K_s K^-1 K_s^T
            # So we compute this for each objective.
            
            cov_mat = np.zeros((len(X_new), self.n_obj, self.n_obj))
            for k in range(self.n_obj):
                # K_s[:, k] is (m, n)
                # Wait, K_s is (m, n) for the kernel.
                # If independent, the kernel is the same for all objectives?
                # Yes, we use the same theta.
                # So K_s is (m, n).
                # v_k = L^-1 K_s (for objective k? No, K_s is the same)
                # Actually, if the kernel is the same, the cross-covariance is the same.
                # So v is (m, n).
                # cov_k = K_ss - v.T @ v
                cov_k = K_ss - v.T @ v
                cov_k = np.maximum(cov_k, 0) # Numerical stability
                cov_mat[:, k, k] = np.diag(cov_k)
                # Off-diagonals are 0
            
            return mean, cov_mat

        # EHVI Calculation
        def ehvi(mean, cov, F_data, ref):
            # mean: (n, 3)
            # cov: (n, 3, 3)
            # F_data: (m, 3)
            # ref: (3,)
            
            # Sample from GP
            n_samples = 20
            # Cholesky of cov for each point?
            # Cov is diagonal (independent objectives).
            # So we can sample independently.
            
            # Mean HV of current set
            hv_current = hypervolume_3d(F_data, ref)
            
            # For each candidate i
            ehvis = np.zeros(len(mean))
            for i in range(len(mean)):
                # Sample f_i
                # std = sqrt(diag(cov[i]))
                std = np.sqrt(np.maximum(np.diag(cov[i]), 0))
                samples = np.random.normal(mean[i], std, size=(n_samples, self.n_obj))
                
                # Calculate HV for each sample
                hv_samples = 0.0
                for s in range(n_samples):
                    F_new = np.vstack([F_data, samples[s]])
                    hv_new = hypervolume_3d(F_new, ref)
                    hv_samples += hv_new
                
                hv_samples /= n_samples
                ehvis[i] = hv_samples - hv_current
                if ehvis[i] < 0:
                    ehvis[i] = 0.0
            return ehvis

        # Main Loop
        while used < self.budget and time.time() < deadline:
            # 1. Generate Candidates
            n_cands = 100
            X_cands = np.random.uniform(lo, hi, size=(n_cands, self.dim))
            
            # 2. GP Predict
            mean, cov = gp_predict(X_cands, X_data, F_data, theta)
            
            # 3. EHVI
            ehvis = ehvi(mean, cov, F_data, ref_point)
            
            # 4. Select Top 10
            top_idx = np.argsort(ehvis)[::-1][:batch_size]
            X_new = X_cands[top_idx]
            
            # 5. Evaluate
            F_new = np.array([problem(x) for x in X_new])
            used += len(X_new)
            
            # 6. Update Data
            X_data = np.vstack([X_data, X_new])
            F_data = np.vstack([F_data, F_new])
            all_F = np.vstack([all_F, F_new])
            
            # 7. Update Hyperparameters (Optional, skip for speed/simplicity)
            # Could do a few steps of gradient ascent on log-likelihood.
            # Given time constraints, we'll keep theta fixed or do a very simple update.
            # Let's keep it fixed for now.

        return X_data
# EVOLVE-BLOCK-END
