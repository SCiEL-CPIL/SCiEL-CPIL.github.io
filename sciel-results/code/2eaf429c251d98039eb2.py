"""Bayesian-Optimization + L-BFGS-B optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """GP-based Bayesian Optimization with L-BFGS-B refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 285s to allow for cleanup/return
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Initial Exploration with structured points
        # Use a mix of center, axis-aligned, and random points
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Axis-aligned points
        for d in range(dim):
            for v in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_points.append(pt)
        # Some random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4.0, 4.0, size=dim))
        
        for x in initial_points:
            if not time_ok():
                break
            eval_and_track(x)

        # Phase 2: Bayesian Optimization with GP surrogate
        # Collect observations
        X_obs = []
        y_obs = []
        
        # Re-evaluate initial points to collect observations (they're already evaluated, 
        # but we need to track them for the GP). Actually, we should track as we go.
        # Let's refactor: track all evaluations in X_obs and y_obs.
        # But eval_and_track doesn't return the point. Let's modify the approach.
        
        # Let's restart the tracking to include all points.
        # Actually, it's easier to just collect points we've evaluated.
        # Since we already evaluated the initial points, let's assume we have them.
        # But we didn't store them. Let's redo the initial phase with tracking.
        
        # Clear and redo
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        X_obs = []
        y_obs = []

        def eval_and_track_with_store(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                X_obs.append(x_clipped.copy())
                y_obs.append(val)
                return val
            except Exception:
                return best_f

        # Phase 1: Initial Exploration
        initial_points = []
        initial_points.append(np.zeros(dim))
        for d in range(dim):
            for v in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_points.append(pt)
        for _ in range(10):
            initial_points.append(rng.uniform(-4.0, 4.0, size=dim))
        
        for x in initial_points:
            if not time_ok():
                break
            eval_and_track_with_store(x)

        # Phase 2: Bayesian Optimization loop
        # Use Expected Improvement (EI) to select next points
        # GP model: y = f(x) + noise, assume small noise
        
        def gp_predict(X_test, X_train, y_train, length_scale=1.0, variance=1.0, noise=0.01):
            """Simple GP prediction using RBF kernel."""
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            
            # Compute kernel matrix K(X_train, X_train)
            def rbf_kernel(X1, X2, length_scale):
                # X1: (n1, d), X2: (n2, d)
                # Compute squared distances
                X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
                X2_sq = np.sum(X2**2, axis=1)[:, np.newaxis].T
                sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-sq_dists / (2 * length_scale**2))
            
            K = rbf_kernel(X_train, X_train, length_scale)
            K += noise * np.eye(n_train)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                # Fallback to lstsq
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
            
            # Predict at X_test
            K_star = rbf_kernel(X_test, X_train, length_scale)
            
            # Mean prediction
            mu = K_star @ alpha
            
            # Variance prediction
            v = np.linalg.solve(L, K_star.T)
            var = variance - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        def expected_improvement(x, best_val, mu, var, length_scale=1.0):
            """Compute Expected Improvement for a point."""
            sigma = np.sqrt(var)
            if sigma < 1e-10:
                return 0.0
            z = (best_val - mu) / sigma
            ei = (best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # BO loop
        max_bo_iter = 30
        for bo_iter in range(max_bo_iter):
            if not time_ok():
                break
            
            # If we have too few observations, just sample random points
            if len(X_obs) < 10:
                x_next = rng.uniform(-4.0, 4.0, size=dim)
            else:
                # Use GP to select next point via EI
                # Evaluate EI on a candidate set
                # Use a grid or random candidates
                n_candidates = 500
                candidates = rng.uniform(-4.0, 4.0, size=(n_candidates, dim))
                
                # Predict mean and variance at candidates
                mu, var = gp_predict(candidates, X_obs, y_obs, length_scale=1.5)
                
                # Compute EI
                ei = np.array([
                    expected_improvement(c, best_f, mu[i], var[i])
                    for i, c in enumerate(candidates)
                ])
                
                # Select best candidate
                x_next = candidates[np.argmax(ei)]
            
            # Evaluate the selected point
            eval_and_track_with_store(x_next)
            
            # Occasionally run a local refinement to escape local optima
            if bo_iter % 5 == 4 and time_ok() and calls < budget - 10:
                try:
                    res = minimize(
                        eval_and_track_with_store,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Phase 3: Final Local Refinement
        if time_ok() and calls < budget - 5:
            try:
                res_lbfgs = minimize(
                    eval_and_track_with_store,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxls': 30}
                )
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x
            except Exception:
                pass

        # Phase 4: Nelder-Mead Polish
        if time_ok() and calls < budget:
            try:
                res_nm = minimize(
                    eval_and_track_with_store,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
