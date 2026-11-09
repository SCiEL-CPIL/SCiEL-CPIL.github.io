"""Hybrid GP-guided Bayesian Optimization with Adaptive Local Refinement for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """
    Optimizer using Gaussian Process (GP) surrogate and Expected Improvement (EI) acquisition.
    Combines global exploration with adaptive local refinement around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        X = []
        Y = []
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to evaluate a point
        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x_clipped))
            except Exception:
                return None
            calls_made += 1
            X.append(x_clipped)
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial Exploration: Space-filling + Random
        n_initial = min(20, budget)
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Axis-aligned points to capture scale
        for i in range(dim):
            for val in [-4.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                initial_points.append(pt)
        
        # Fill remaining with random
        n_rand = n_initial - len(initial_points)
        if n_rand > 0:
            initial_points.extend(rng.uniform(-5, 5, (n_rand, dim)).tolist())
        
        for pt in initial_points:
            if time.time() > deadline or calls_made >= budget:
                break
            evaluate(pt)

        # GP Prediction Helper
        def gp_predict(X_train, Y_train, X_test, length_scale, sigma_f, noise=1e-6):
            X_train = np.atleast_2d(X_train)
            X_test = np.atleast_2d(X_test)
            
            # Normalize Y
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train)
            if Y_std < 1e-10:
                Y_std = 1.0
            Y_norm = (Y_train - Y_mean) / Y_std

            def kernel(X1, X2, ls, sf):
                # Squared Exponential Kernel
                # d2 = ||x1 - x2||^2
                d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
                d2 = np.maximum(d2, 0)
                return (sf**2) * np.exp(-d2 / (2 * ls**2))

            K = kernel(X_train, X_train, length_scale, sigma_f)
            K[np.diag_indices_from(K)] += noise
            K_s = kernel(X_train, X_test, length_scale, sigma_f)

            try:
                # Use Cholesky for stability if possible, else solve
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                mu = K_s.T @ alpha
                
                # Variance
                v = np.linalg.solve(L, K_s)
                var = sigma_f**2 - np.sum(K_s**2 * v, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to regular solve
                try:
                    alpha = np.linalg.solve(K, Y_norm)
                    mu = K_s.T @ alpha
                    v = np.linalg.solve(K, K_s)
                    var = sigma_f**2 - np.sum(K_s**2 * v, axis=0)
                    var = np.maximum(var, 1e-10)
                except np.linalg.LinAlgError:
                    mu = np.full(X_test.shape[0], Y_mean)
                    var = np.full(X_test.shape[0], Y_std**2)

            mu = mu * Y_std + Y_mean
            var = var * Y_std**2
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, best_y, xi=0.0):
            s = np.sqrt(var)
            z = (best_y - mu - xi) / (s + 1e-10)
            ei = (best_y - mu - xi) * norm.cdf(z) + s * norm.pdf(z)
            return np.maximum(ei, 0)

        # Main Optimization Loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Adaptive Hyperparameters
            if len(X) >= 5:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                
                # Estimate length scale from data spread
                # Use median distance between points as a heuristic for local scale
                # Or simply use the standard deviation of coordinates
                data_spread = np.std(X_arr, axis=0)
                avg_spread = np.mean(data_spread[data_spread > 1e-4]) if np.any(data_spread > 1e-4) else 1.0
                # Weierstrass is oscillatory, so length scale should be relatively small
                length_scale = max(0.2, min(avg_spread * 0.5, 2.0))
                
                sigma_f = max(np.std(Y_arr), 0.1)
            else:
                length_scale = 1.0
                sigma_f = 1.0

            # Generate Candidates
            n_candidates = 100
            candidates = []
            
            # 1. Global Random (Exploration)
            n_global = n_candidates // 3
            candidates.append(rng.uniform(-5, 5, (n_global, dim)))
            
            # 2. Local Refinement (Exploitation)
            # Adaptive sigma: starts wide, narrows down
            progress = calls_made / budget
            local_sigma = 2.0 * (1.0 - progress) + 0.1
            if best_x is not None:
                n_local = n_candidates // 3
                local_cands = best_x + rng.standard_normal((n_local, dim)) * local_sigma
                local_cands = np.clip(local_cands, -5, 5)
                candidates.append(local_cands)
            
            # 3. GP-Guided Random (Balanced)
            n_gp = n_candidates - n_global - (n_local if best_x is not None else 0)
            if n_gp > 0:
                gp_cands = rng.uniform(-5, 5, (n_gp, dim))
                candidates.append(gp_cands)

            all_cands = np.vstack(candidates)

            # Select points using EI
            n_select = min(5, budget - calls_made) # Evaluate 5 points per iteration
            
            if len(X) >= 5:
                mu, var = gp_predict(np.array(X), np.array(Y), all_cands, length_scale, sigma_f)
                ei = expected_improvement(mu, var, best_f, xi=0.01)
                
                # To ensure diversity, we can pick top EI but also force some random if EI is low
                top_idx = np.argsort(ei)[::-1][:n_select]
                selected = all_cands[top_idx]
            else:
                selected = all_cands[:n_select]

            # Evaluate selected points
            for pt in selected:
                if calls_made >= budget or time.time() > deadline:
                    break
                evaluate(pt)

        return best_x, best_f
# EVOLVE-BLOCK-END
