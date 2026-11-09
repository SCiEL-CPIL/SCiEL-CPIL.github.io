"""Rewritten optimizer: surrogate-assisted multi-objective search with Sobol init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining Sobol initialization,
    Gaussian Process surrogate with EHVI acquisition, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (from task description)
        ref = np.array([3000.0, 0.06])

        # --- Sobol quasi-random initialization ---
        n_init = min(64, budget // 3)
        # Use scipy's qmc for Sobol sequence
        from scipy.integrate import quad
        from scipy.stats import norm
        try:
            from scipy.integrate import quad
            # Sobol via simple LHS as fallback if qmc not available
            try:
                import scipy.stats as stats
                # Use scipy.stats.qmc if available (SciPy >= 1.7)
                from scipy.stats import qmc
                sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
                samples = sobol.random(n=n_init)
            except ImportError:
                # Fallback: Latin Hypercube Sampling
                rng = np.random.default_rng(42)
                samples = np.zeros((n_init, dim))
                for d in range(dim):
                    perm = rng.permutation(n_init)
                    samples[:, d] = (perm + rng.random(n_init)) / n_init
        except Exception:
            rng = np.random.default_rng(42)
            samples = rng.random((n_init, dim))

        X = lo + (hi - lo) * samples
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points and their objectives
        all_X = [X.copy()]
        all_F = [F.copy()]

        def time_ok():
            return time.time() < deadline

        def non_dominated_indices(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def hypervolume_2d(points, ref_point):
            """Exact hypervolume for 2D points dominated by ref_point."""
            pts = [p for p in points if np.all(p < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
                else:
                    hv += (ref_point[0] - p[0]) * (ref_point[1] - p[1])
            return hv

        def ehvi_score(x, F_train, mu_pred, sigma_pred, ref_pt):
            """Compute EHVI for a candidate x given GP predictions."""
            # F_train: (n, n_obj), mu_pred: (n_obj,), sigma_pred: (n_obj,)
            # Simplified EHVI: expected improvement in hypervolume
            # For 2D, we can approximate
            n = len(F_train)
            if n == 0:
                return 0.0
            
            # Current hypervolume of non-dominated set
            nd_idx = non_dominated_indices(F_train)
            nd_pts = F_train[nd_idx]
            current_hv = hypervolume_2d(nd_pts, ref_pt)
            
            # Expected hypervolume if x is added
            # Approximate: use mean prediction
            new_pts = np.vstack([nd_pts, mu_pred])
            new_nd_idx = non_dominated_indices(new_pts)
            new_nd_pts = new_pts[new_nd_idx]
            new_hv = hypervolume_2d(new_nd_pts, ref_pt)
            
            # EHVI ~ max(0, new_hv - current_hv) adjusted by uncertainty
            improvement = max(0.0, new_hv - current_hv)
            # Add exploration bonus
            exploration = np.sum(sigma_pred) * 0.1
            return improvement + exploration

        # --- GP-based acquisition loop ---
        # We'll use a simple GP with RBF kernel for each objective
        def gp_predict(X_train, y_train, X_test, length_scale=0.3):
            """Simple GP prediction with RBF kernel."""
            n = len(X_train)
            d = X_train.shape[1]
            # Standardize
            mu_x = X_train.mean(axis=0)
            std_x = X_train.std(axis=0) + 1e-8
            X_train_s = (X_train - mu_x) / std_x
            X_test_s = (X_test - mu_x) / std_x
            
            # RBF kernel
            def rbf(A, B):
                sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                return np.exp(-0.5 * sq / length_scale**2)
            
            K = rbf(X_train_s, X_train_s) + 1e-6 * np.eye(n)
            K_y = y_train - y_train.mean()
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, K_y))
            except np.linalg.LinAlgError:
                return y_train.mean() * np.ones(len(X_test)), np.ones(len(X_test)) * y_train.std()
            
            K_s = rbf(X_test_s, X_train_s)
            mu = y_train.mean() + K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = 1.0 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            sigma = np.sqrt(var)
            return mu, sigma

        # Main search loop
        while used < budget and time_ok():
            # Get non-dominated points from all evaluations
            all_F_flat = np.vstack(all_F)
            all_X_flat = np.vstack(all_X)
            nd_idx = non_dominated_indices(all_F_flat)
            nd_X = all_X_flat[nd_idx]
            nd_F = all_F_flat[nd_idx]
            
            # If we have enough points, use GP for acquisition
            if len(nd_F) >= 4:
                # Use GP on the full training set for better generalization
                train_X = all_X_flat
                train_F = all_F_flat
                
                # Sample candidate points for acquisition
                n_candidates = 200
                rng = np.random.default_rng(used)
                candidates = lo + (hi - lo) * rng.random((n_candidates, dim))
                
                # Evaluate EHVI for each candidate
                best_cand = None
                best_score = -np.inf
                for cand in candidates:
                    mu, sigma = gp_predict(train_X, train_F, cand.reshape(1, -1))
                    mu = mu[0]
                    sigma = sigma[0]
                    score = ehvi_score(cand, nd_F, mu, sigma, ref)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand is not None:
                    x_new = best_cand
                else:
                    x_new = lo + (hi - lo) * rng.random(dim)
            else:
                # Random exploration when we have few points
                rng = np.random.default_rng(used)
                x_new = lo + (hi - lo) * rng.random(dim)
            
            f_new = problem(x_new)
            used += 1
            all_X.append(x_new.reshape(1, -1))
            all_F.append(np.array([f_new]))
        
        # --- Local refinement around best points ---
        if time_ok() and used < budget:
            all_F_flat = np.vstack(all_F)
            all_X_flat = np.vstack(all_X)
            nd_idx = non_dominated_indices(all_F_flat)
            nd_X = all_X_flat[nd_idx]
            
            # Refine top few non-dominated points with local optimization
            top_k = min(5, len(nd_X))
            for i in range(top_k):
                if not time_ok() or used >= budget:
                    break
                x0 = nd_X[i]
                # Local perturbation
                rng = np.random.default_rng(used + i)
                for _ in range(3):
                    if not time_ok() or used >= budget:
                        break
                    x_try = x0 + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = problem(x_try)
                    used += 1
                    all_X.append(x_try.reshape(1, -1))
                    all_F.append(np.array([f_try]))
                    # Update x0 if better
                    if np.all(f_try <= nd_F[i]):
                        x0 = x_try
                        nd_F[i] = f_try
                        nd_X[i] = x0
        
        # Return the best non-dominated set
        all_F_final = np.vstack(all_F)
        all_X_final = np.vstack(all_X)
        nd_idx_final = non_dominated_indices(all_F_final)
        return all_X_final[nd_idx_final]

# EVOLVE-BLOCK-END
