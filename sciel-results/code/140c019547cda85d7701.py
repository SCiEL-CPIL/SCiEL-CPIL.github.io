"""Surrogate-assisted multi-objective optimizer using GP + EHVI for RE34."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Gaussian Process surrogate-assisted multi-objective optimizer using EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random.default_rng(42)

        # Initial random sampling
        n_init = min(25, self.budget)
        X = lo + (hi - lo) * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # GP kernel parameters (shared across objectives for simplicity)
        # We'll use a simple RBF kernel with length scales per dimension
        lengthscales = np.ones(self.dim) * 0.5
        variance = 1.0
        noise = 1e-4

        def gp_predict(X_train, F_train, X_test, obj_idx):
            """Predict mean and variance for one objective at X_test."""
            y = F_train[:, obj_idx]
            n = len(y)
            
            # Standardize y for better GP conditioning
            y_mean = y.mean()
            y_std = y.std() + 1e-8
            y_std = y_std if y_std > 1e-8 else 1.0
            y_norm = (y - y_mean) / y_std
            
            # Compute kernel matrix
            K = self._rbf_kernel(X_train, X_train, lengthscales) + noise * np.eye(n)
            K_test = self._rbf_kernel(X_train, X_test, lengthscales)
            
            # Solve
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            mu_norm = K_test.T @ alpha
            v = np.linalg.solve(L, K_test)
            var = variance - (v ** 2).sum(axis=0)
            var = np.maximum(var, 1e-10)
            
            # Unstandardize
            mu = mu_norm * y_std + y_mean
            var = var * (y_std ** 2)
            
            return mu, var

        def _rbf_kernel(self, X1, X2, lengthscales):
            """RBF kernel."""
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            sq_dists = ((X1_scaled ** 2).sum(axis=1)[:, None] +
                        (X2_scaled ** 2).sum(axis=1)[None, :] -
                        2 * X1_scaled @ X2_scaled.T)
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-0.5 * sq_dists)

        # Pareto front computation
        def compute_pareto_front(F):
            """Return indices of non-dominated points."""
            n = len(F)
            is_pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_pareto[i]:
                    continue
                for j in range(n):
                    if i == j or not is_pareto[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_pareto[i] = False
                        break
            return np.where(is_pareto)[0]

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points dominated by ref."""
            points = [p for p in points if np.all(p < ref)]
            if not points:
                return 0.0
            points = sorted(points, key=lambda p: p[0])
            vol = 0.0
            prev_y = ref[1]
            for p in points:
                vol += (ref[0] - p[0]) * (prev_y - p[1])
                prev_y = min(prev_y, p[1])
            return vol

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume using inclusion-exclusion / partitioning."""
            # Filter points dominated by ref
            points = [p for p in points if np.all(p < ref)]
            if not points:
                return 0.0
            
            # Use a simple approach: sort by first objective, use 2D HV for each slice
            # This is an approximation but works for small sets
            # Better: use the standard 3D HV algorithm
            
            # For small number of points, use a grid-based or partitioning approach
            # Here we use a recursive partitioning method
            
            def hv_3d(pts, ref):
                if not pts:
                    return 0.0
                # Sort by first coordinate
                pts = sorted(pts, key=lambda p: p[0])
                vol = 0.0
                prev_x = ref[0]
                for p in pts:
                    # For this point, the contribution is (ref[0]-p[0]) * HV_2d of remaining points in 2D
                    # Actually, we need to be more careful
                    # The hypervolume contributed by p is (ref[0]-p[0]) * (area of 2D region dominated by p in dims 1,2)
                    # but only the part not already counted
                    # Simpler: just compute it properly
                    pass
                
                # Use a different approach: enumerate all hypercubes
                # For n points in 3D, the number of cells is O(n^3) which is fine for small n
                xs = sorted(set([ref[0]] + [p[0] for p in pts]))
                ys = sorted(set([ref[1]] + [p[1] for p in pts]))
                zs = sorted(set([ref[2]] + [p[2] for p in pts]))
                
                total = 0.0
                for i in range(len(xs) - 1):
                    for j in range(len(ys) - 1):
                        for k in range(len(zs) - 1):
                            # Cell [xs[i], xs[i+1]] x [ys[j], ys[j+1]] x [zs[k], zs[k+1]]
                            # Check if this cell is dominated by any point
                            cell_center = np.array([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
                            # A cell is covered if there exists a point p such that p <= cell_corner (the far corner)
                            # Actually, a cell is in the hypervolume if it's dominated by at least one point
                            # The cell is dominated if there's a point p with p[0] <= xs[i], p[1] <= ys[j], p[2] <= zs[k]
                            # Wait, no. The hypervolume is the union of boxes from each point to ref.
                            # A point p dominates the region [p, ref].
                            # So cell [xs[i],xs[i+1]]x[ys[j],ys[j+1]]x[zs[k],zs[k+1]] is in HV if
                            # there exists p with p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]
                            dominated = False
                            for p in pts:
                                if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                    dominated = True
                                    break
                            if dominated:
                                total += (xs[i+1]-xs[i]) * (ys[j+1]-ys[j]) * (zs[k+1]-zs[k])
                return total
            
            return hv_3d(points, ref)

        def ehvi_score(X_candidate, X_train, F_train, ref):
            """Compute EHVI for a candidate point."""
            n_obj = self.n_obj
            # Predict means and variances for all objectives
            mus = np.zeros(n_obj)
            vars_ = np.zeros(n_obj)
            for o in range(n_obj):
                mu, var = gp_predict(X_train, F_train, X_candidate.reshape(1, -1), o)
                mus[o] = mu[0]
                vars_[o] = var[0]
            
            # Compute EHVI using Monte Carlo or analytical for 3D
            # For 3 objectives, analytical EHVI is complex, so use MC
            n_mc = 100
            rng_ehvi = np.random.default_rng(0)
            samples = rng_ehvi.normal(mus, np.sqrt(vars_), size=(n_mc, n_obj))
            
            # Current Pareto front
            pareto_idx = compute_pareto_front(F_train)
            current_front = F_train[pareto_idx]
            
            # Compute HV of current front
            current_hv = hypervolume_3d(current_front, ref)
            
            # Compute HV with new sample
            hv_improvements = []
            for s in samples:
                if np.any(s >= ref):
                    hv_improvements.append(0.0)
                    continue
                new_front = np.vstack([current_front, s.reshape(1, -1)])
                new_pareto_idx = compute_pareto_front(new_front)
                new_front = new_front[new_pareto_idx]
                new_hv = hypervolume_3d(new_front, ref)
                hv_improvements.append(max(0, new_hv - current_hv))
            
            return np.mean(hv_improvements)

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points using different strategies
            candidates = []
            
            # 1. EHVI-based candidates (sample in the space)
            n_ehvi = 20
            for _ in range(n_ehvi):
                x_cand = lo + (hi - lo) * rng.random(self.dim)
                candidates.append(x_cand)
            
            # 2. Local refinement around best Pareto points
            pareto_idx = compute_pareto_front(all_F)
            if len(pareto_idx) > 0:
                n_local = 10
                for _ in range(n_local):
                    p_idx = pareto_idx[rng.integers(len(pareto_idx))]
                    x_base = all_X[p_idx]
                    x_cand = x_base + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)
            
            # 3. Random exploration
            n_random = 5
            for _ in range(n_random):
                x_cand = lo + (hi - lo) * rng.random(self.dim)
                candidates.append(x_cand)
            
            # Score candidates using EHVI (subset to save time)
            n_score = min(30, len(candidates))
            idx_score = rng.choice(len(candidates), size=n_score, replace=False)
            
            scores = []
            for idx in idx_score:
                x_cand = candidates[idx]
                # Check if already evaluated
                dists = np.linalg.norm(all_X - x_cand, axis=1)
                if np.min(dists) < 1e-6:
                    scores.append(-np.inf)
                    continue
                score = ehvi_score(x_cand, all_X, all_F, ref)
                scores.append(score)
            
            scores = np.array(scores)
            best_local_idx = idx_score[np.argmax(scores)]
            x_new = candidates[best_local_idx]
            
            # Evaluate
            f_new = problem(x_new)
            used += 1
            all_X = np.vstack([all_X, x_new.reshape(1, -1)])
            all_F = np.vstack([all_F, np.array(f_new).reshape(1, -1)])
        
        return all_X
# EVOLVE-BLOCK-END
