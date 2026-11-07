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

        # Initial random sampling
        n_init = min(30, self.budget)
        X = lo + (hi - lo) * np.random.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # GP hyperparameters
        alpha = 1e-4  # noise
        length_scales = np.ones(self.dim) * 0.5
        signal_var = 1.0

        def compute_K(X1, X2, length_scales, signal_var):
            """Compute GP kernel matrix with RBF kernel."""
            X1s = X1 / length_scales
            X2s = X2 / length_scales
            sq_dists = np.sum(X1s ** 2, axis=1)[:, None] + np.sum(X2s ** 2, axis=1)[None, :] - 2 * X1s @ X2s.T
            sq_dists = np.maximum(sq_dists, 0)
            K = signal_var ** 2 * np.exp(-0.5 * sq_dists)
            return K

        def get_nondominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(n):
                    if i == j or is_dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        def hypervolume_3d(points, ref_point):
            """Calculate hypervolume for 3 objectives."""
            if len(points) == 0:
                return 0.0
            # Filter points dominated by ref_point
            valid = points[~np.any(points >= ref_point, axis=1)]
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective
            valid = valid[np.argsort(valid[:, 0])]
            
            # Iterate and calculate volume
            hv = 0.0
            prev_y = ref_point[1]
            prev_z = ref_point[2]
            
            # We need to be careful with the sorting and calculation for 3D.
            # A simpler approach for small N:
            # Sort by obj 0
            pts = valid.copy()
            pts = pts[np.argsort(pts[:, 0])]
            
            # We can use a sweep line or just sum contributions.
            # For 3D, we can sort by obj 0, then for each point, find the "envelope" of the others.
            # Actually, for small N (< 100), we can just use the inclusion-exclusion or simple decomposition.
            
            # Let's use a simple algorithm:
            # Sort by first dimension.
            # For each point, the volume contributed is the volume of the box from this point to ref,
            # minus the volume already covered by points with smaller first dimension.
            # This is tricky. Let's use a robust O(N^2) or O(N log N) method.
            
            # Method:
            # 1. Sort by f0.
            # 2. Keep track of the "maximal" f1 and f2 for the remaining points.
            # Actually, let's just use a standard 3D HV algorithm.
            
            # Sort by f0 ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # We'll maintain a list of "active" points that are not dominated in the (f1, f2) space
            # by any point with a smaller f0.
            active = []
            hv = 0.0
            
            for p in pts:
                f0, f1, f2 = p
                # Remove points from active that are dominated by p in (f1, f2)
                # A point q in active is dominated by p if f1_p <= f1_q and f2_p <= f2_q
                new_active = []
                for q in active:
                    if f1 <= q[1] and f2 <= q[2]:
                        # q is dominated by p, remove it
                        continue
                    new_active.append(q)
                active = new_active
                active.append((f1, f2))
                
                # Calculate HV contribution of p
                # The contribution is the volume of the box [f0, ref0] x [f1, ref1] x [f2, ref2]
                # minus the volume of boxes formed by active points that are "inside" this box.
                # This is complex. Let's use a simpler approximation or a library if available? No library.
                # Let's use the fact that N is small.
                
            # Alternative: Calculate HV of the set of non-dominated points directly.
            # Since we already have non-dominated points, we can use a simple recursive or iterative method.
            # For 3D, we can project onto 2D and use 2D HV logic?
            
            # Let's implement a simple O(N^2) HV for 3D.
            # Sort by f0.
            # For each point i, the volume it contributes is:
            # (ref0 - f0_i) * (volume of the region in (f1, f2) space dominated by i and not dominated by any j with f0_j < f0_i)
            
            # Let's try a different approach:
            # 1. Sort points by f0.
            # 2. Maintain a "front" in (f1, f2) space.
            # 3. For each new point, calculate the area in (f1, f2) that it dominates which is not already dominated by previous points.
            
            pts = valid.copy()
            pts = pts[np.argsort(pts[:, 0])]
            
            # We will maintain a list of segments in the (f1, f2) plane?
            # Actually, for small N, we can just check dominance against the "current best" front in 2D.
            
            # Let's use a simpler method:
            # HV = sum over all points i of (ref0 - f0_i) * A_i
            # where A_i is the area in (f1, f2) space that is dominated by i and not dominated by any point j with f0_j <= f0_i.
            
            # This is still complex. Let's use a Monte Carlo approximation if exact is too slow?
            # No, N is small. Let's use a standard algorithm.
            
            # Algorithm:
            # 1. Sort by f0.
            # 2. Keep a list of non-dominated points in (f1, f2) space.
            # 3. For each point, calculate the area of the (f1, f2) box it dominates that is not covered by the current (f1, f2) front.
            
            # Let's implement 2D HV calculation for the "shadow" of the points.
            # This is getting complicated. Let's use a simple sampling-based HV estimator for the candidate selection,
            # and exact HV only for the final score? No, we need EHVI.
            
            # Let's use a simple O(N^2) HV calculation for 3D.
            # It's sufficient for N < 100.
            
            # We'll use the "layer" approach.
            # Sort by f0.
            # For each point, determine the "new" volume it adds.
            
            # Let's just use a simple recursive definition or a known implementation.
            # Here is a simple O(N^2) implementation for 3D:
            
            # Sort by f0
            pts = pts[np.argsort(pts[:, 0])]
            
            # We maintain a list of "active" points in (f1, f2) that form the non-dominated front.
            # Actually, we can just calculate the HV of the set of non-dominated points.
            # We already have the non-dominated points from get_nondominated.
            
            # Let's just use a simple Monte Carlo for HV inside EHVI?
            # No, EHVI needs accurate HV.
            
            # Let's use a standard 3D HV algorithm.
            # Reference: "A Fast Hypervolume Calculation for Multi-Objective Optimization"
            
            # For now, let's use a simple approximation:
            # HV ≈ sum of volumes of boxes from each point to ref, subtracting overlaps.
            # This is hard.
            
            # Let's use the fact that we can calculate HV of a set of non-dominated points in 3D
            # by sorting by f0 and then calculating the 2D HV of the (f1, f2) projections
            # for each "slice".
            
            # Let's implement a simple 2D HV first.
            def hv_2d(points_2d, ref_2d):
                if len(points_2d) == 0:
                    return 0.0
                # Sort by x
                pts = points_2d[np.argsort(points_2d[:, 0])]
                hv = 0.0
                max_y = -np.inf
                prev_x = ref_2d[0]
                for p in pts:
                    x, y = p
                    if y > max_y:
                        hv += (ref_2d[0] - x) * (ref_2d[1] - y)
                        max_y = y
                return hv

            # 3D HV:
            # Sort by f0.
            # We can decompose the 3D HV into a sum of 2D HVs weighted by the difference in f0.
            # This is not straightforward.
            
            # Let's use a simple Monte Carlo for HV in the EHVI loop.
            # Since we are evaluating many candidates, speed is key.
            # 1000 samples for HV estimation should be fast enough.
            
            n_hv_samples = 1000
            rng_hv = np.random.RandomState(123)
            # Sample points in the box [min_f, ref]
            # We need to estimate the volume of the region dominated by the points.
            # This is equivalent to the volume of the union of boxes.
            
            # Let's just use the non-dominated points and calculate HV exactly using a simple O(N^2) method.
            # For 3D, we can sort by f0, and for each point, find the "new" area in (f1, f2) it covers.
            
            # Let's implement a simple O(N^2) 3D HV.
            # Sort by f0
            pts = valid[np.argsort(valid[:, 0])]
            n = len(pts)
            hv = 0.0
            
            # We will maintain a list of "active" points in (f1, f2) space.
            # Actually, we can just check for each point if it is dominated by any other point.
            # We already filtered non-dominated points.
            
            # Let's use a simple algorithm:
            # For each point i, the volume it contributes is:
            # (ref0 - f0_i) * Area_i
            # where Area_i is the area in (f1, f2) space that is dominated by i and not dominated by any j with f0_j < f0_i.
            
            # We can calculate Area_i by checking against all j with f0_j < f0_i.
            
            for i in range(n):
                f0_i, f1_i, f2_i = pts[i]
                
                # Find the "envelope" of points j < i in (f1, f2) space.
                # The area dominated by i is the area of the box [f1_i, ref1] x [f2_i, ref2]
                # minus the area covered by the union of boxes of j < i.
                
                # This is complex. Let's use a simpler approximation:
                # Assume the points are well-distributed.
                # HV ≈ sum (ref0 - f0_i) * (ref1 - f1_i) * (ref2 - f2_i) / N? No.
                
                # Let's just use Monte Carlo for HV.
                # It's robust and fast enough for N < 100.
                
            # Let's implement Monte Carlo HV.
            # Sample points in the bounding box [min_f, ref].
            # Count how many are dominated by at least one point in the set.
            
            min_f = np.min(pts, axis=0)
            # Sample points
            samples = rng_hv.random_sample((n_hv_samples, 3))
            samples = min_f + samples * (ref_point - min_f)
            
            # Check dominance
            # A sample is dominated if there exists a point p such that p <= sample in all objectives.
            # Vectorized check:
            # p <= sample for all j
            # shape: (n, 3) and (n_hv_samples, 3)
            # We can do this by broadcasting.
            # But n * n_hv_samples * 3 might be large.
            # n ~ 50, n_hv_samples = 1000. 50 * 1000 * 3 = 150,000 operations. Fast.
            
            # Let's do it.
            dominated = np.zeros(n_hv_samples, dtype=bool)
            for p in pts:
                # If p <= sample, then sample is dominated by p.
                # We only need to check if p is better than sample in all objectives.
                # p <= sample
                mask = np.all(p <= samples, axis=1)
                dominated |= mask
                if np.all(dominated):
                    break
            
            volume_box = np.prod(ref_point - min_f)
            hv = np.sum(dominated) / n_hv_samples * volume_box
            return hv

        def ehvi_candidate(X_new, X_train, F_train, ref_point, length_scales, signal_var, alpha):
            """Compute EHVI for a candidate point."""
            # Predict each objective
            means = []
            stds = []
            for j in range(self.n_obj):
                y_j = F_train[:, j]
                # Use precomputed Cholesky if possible, but for simplicity, recompute
                # This is slow. Let's optimize.
                K = compute_K(X_train, X_train, length_scales, signal_var)
                K[np.diag_indices_from(K)] += alpha
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K[np.diag_indices_from(K)] += 1e-6
                    L = np.linalg.cholesky(K)
                
                K_s = compute_K(X_new, X_train, length_scales, signal_var)
                # Solve L L^T x = y
                # x = L^T \ (L \ y)
                sol1 = np.linalg.solve(L, y_j)
                sol2 = np.linalg.solve(L.T, sol1)
                mean = K_s @ sol2
                
                # Variance
                # v = L \ (L^T \ K_s.T)
                # var = K_ss - v @ v.T
                v1 = np.linalg.solve(L, K_s.T)
                v2 = np.linalg.solve(L.T, v1)
                K_ss = compute_K(X_new, X_new, length_scales, signal_var)
                var = K_ss[np.diag_indices_from(K_ss)] - np.sum(v2 ** 2, axis=1)
                var = np.maximum(var, 0)
                std = np.sqrt(var)
                
                means.append(mean)
                stds.append(std)
            
            means = np.array(means)  # shape (n_obj,)
            stds = np.array(stds)    # shape (n_obj,)

            # EHVI approximation via sampling
            n_samples = 100
            rng_local = np.random.RandomState(42)
            # Sample from predictive distribution
            samples = means + stds * rng_local.standard_normal((n_samples, self.n_obj))
            
            # Compute hypervolume contribution
            hv_improvements = []
            # Get current non-dominated front
            front_indices = get_nondominated(F_train)
            F_front = F_train[front_indices]
            
            for s in samples:
                if np.any(s >= ref_point):
                    continue
                
                # Check if s is dominated by any point in F_front
                dominated = False
                for p in F_front:
                    if np.all(p <= s) and np.any(p < s):
                        dominated = True
                        break
                
                if dominated:
                    hv_improvements.append(0.0)
                    continue
                
                # Calculate HV of F_front U {s}
                # We can just add s to F_front and calculate HV
                F_new = np.vstack([F_front, s[None, :]])
                hv_new = hypervolume_3d(F_new, ref_point)
                hv_old = hypervolume_3d(F_front, ref_point)
                
                hv_improvements.append(max(0, hv_new - hv_old))
            
            ehvi_val = np.mean(hv_improvements)
            return ehvi_val

        # Main optimization loop
        while used < self.budget:
            # Generate candidate points
            n_candidates = min(50, self.budget - used)
            X_cand = []
            # Random candidates
            n_rand = max(10, n_candidates // 2)
            X_cand.append(lo + (hi - lo) * np.random.random((n_rand, self.dim)))
            # Local perturbations around best points
            n_local = n_candidates - n_rand
            if len(X) > 0 and n_local > 0:
                n_perturb = min(10, len(X))
                idx_perturb = np.random.choice(len(X), size=n_perturb, replace=False)
                for idx in idx_perturb:
                    n_per_point = max(1, n_local // n_perturb)
                    perturb = X[idx][None, :] + 0.1 * (hi - lo) * np.random.standard_normal((n_per_point, self.dim))
                    perturb = np.clip(perturb, lo, hi)
                    X_cand.append(perturb)
            
            X_cand = np.vstack(X_cand)
            X_cand = np.clip(X_cand, lo, hi)
            
            # Evaluate EHVI for each candidate
            ehvi_vals = []
            for x_c in X_cand:
                x_c = x_c.reshape(1, -1)
                try:
                    val = ehvi_candidate(x_c, X, F, ref_point, length_scales, signal_var, alpha)
                except Exception:
                    val = 0.0
                ehvi_vals.append(val)
            
            ehvi_vals = np.array(ehvi_vals)
            best_idx = np.argmax(ehvi_vals)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            used += 1
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            
            # Update hyperparameters occasionally
            if used % 20 == 0:
                data_range = hi - lo
                length_scales = np.maximum(length_scales * 1.1, 0.01)
                length_scales = np.minimum(length_scales, data_range * 0.5)
        
        return X
# EVOLVE-BLOCK-END
