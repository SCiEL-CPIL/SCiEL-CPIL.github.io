"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Gaussian Process surrogate with Expected Improvement acquisition,
    followed by L-BFGS-B polish for final precision."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: GP-based Bayesian optimization for global search ---
        # Collect initial samples using a space-filling design
        rng = np.random.RandomState(12345)

        # Latin hypercube for initial design
        n_initial = 20
        X_init = np.zeros((n_initial, d))
        for j in range(d):
            perm = rng.permutation(n_initial)
            X_init[:, j] = (perm + 0.5) / n_initial * 10.0 - 5.0

        # Evaluate initial design
        X_obs = []
        y_obs = []
        for x in X_init:
            if not time_ok() or calls >= budget:
                break
            fx = safe_func(x)
            X_obs.append(x.copy())
            y_obs.append(fx)

        # Add a few extra random points for diversity
        for _ in range(5):
            if not time_ok() or calls >= budget:
                break
            x = rng.uniform(-4, 4, size=d)
            fx = safe_func(x)
            X_obs.append(x.copy())
            y_obs.append(fx)

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # GP hyperparameters
        # Use a Matern 5/2 kernel with automatic relevance determination
        length_scales = np.ones(d) * 2.0
        sigma_f = 1.0  # Signal variance
        sigma_n = 1e-10  # Noise variance

        def matern52_kernel(X1, X2, length_scales, sigma_f):
            """Matern 5/2 kernel."""
            # Scale inputs
            X1_s = X1 / length_scales
            X2_s = X2 / length_scales
            # Compute squared distances
            diff = X1_s[:, np.newaxis, :] - X2_s[np.newaxis, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=2))
            sqrt3_r = np.sqrt(3.0) * r
            K = sigma_f ** 2 * (1.0 + sqrt3_r + sqrt3_r ** 2 / 3.0) * np.exp(-sqrt3_r)
            return K

        def gp_predict(X_new, X_obs, y_obs, length_scales, sigma_f, sigma_n):
            """Predict mean and variance at X_new given observations."""
            K = matern52_kernel(X_obs, X_obs, length_scales, sigma_f)
            K += sigma_n * np.eye(len(X_obs))
            K_s = matern52_kernel(X_obs, X_new, length_scales, sigma_f)
            K_ss = matern52_kernel(X_new, X_new, length_scales, sigma_f)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
                v = np.linalg.solve(L, K_s.T)
                mu = K_s @ alpha
                var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-12)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                try:
                    K_inv = np.linalg.pinv(K)
                    mu = K_s @ K_inv @ y_obs
                    K_ss_full = matern52_kernel(X_new, X_new, length_scales, sigma_f)
                    var = np.diag(K_ss_full) - np.sum(K_s @ K_inv * K_s.T, axis=0)
                    var = np.maximum(var, 1e-12)
                except:
                    mu = np.full(len(X_new), y_obs.mean())
                    var = np.full(len(X_new), y_obs.var() + 1e-6)
            
            return mu, var

        # Expected Improvement acquisition function
        def expected_improvement(mu, var, y_best, xi=0.01):
            """Compute EI for minimization."""
            sigma = np.sqrt(var)
            improvement = y_best - mu - xi
            z = improvement / (sigma + 1e-12)
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Hyperparameter tuning via simple grid search on a subset
        # Try different length scales
        ls_candidates = [0.5, 1.0, 2.0, 3.0, 5.0]
        best_ls = length_scales
        best_score = np.inf

        if len(X_obs) >= 5:
            # Use a validation set approach: hold out last 20%
            n_val = max(2, len(X_obs) // 5)
            X_train = X_obs[:-n_val]
            y_train = y_obs[:-n_val]
            X_val = X_obs[-n_val:]
            
            for ls in ls_candidates:
                ls_vec = np.ones(d) * ls
                try:
                    mu_val, var_val = gp_predict(X_val, X_train, y_train, ls_vec, sigma_f, sigma_n)
                    # Use negative log-likelihood as score
                    var_val = np.maximum(var_val, 1e-12)
                    nll = 0.5 * np.sum(np.log(2 * np.pi * var_val)) + 0.5 * np.sum((y_val - mu_val) ** 2 / var_val)
                    if nll < best_score:
                        best_score = nll
                        best_ls = ls_vec
                except:
                    continue
        
        length_scales = best_ls

        # BO loop: select next point using EI
        n_bo_iter = min(80, budget - calls - 30)  # Reserve 30 calls for L-BFGS-B
        candidate_pool_size = 200

        for bo_iter in range(n_bo_iter):
            if not time_ok() or calls >= budget - 30:
                break

            # Generate candidate points: mix of random and local around best
            candidates = []
            
            # Random candidates
            n_rand = candidate_pool_size // 2
            for _ in range(n_rand):
                x = rng.uniform(-5, 5, size=d)
                candidates.append(x)
            
            # Local candidates around best point
            n_local = candidate_pool_size // 4
            for _ in range(n_local):
                x = best_x + rng.randn(d) * 0.5
                x = np.clip(x, -5, 5)
                candidates.append(x)
            
            # Candidates around second best
            n_local2 = candidate_pool_size // 4
            if len(y_obs) > 1:
                sorted_idx = np.argsort(y_obs)
                second_best_x = X_obs[sorted_idx[1]]
                for _ in range(n_local2):
                    x = second_best_x + rng.randn(d) * 0.8
                    x = np.clip(x, -5, 5)
                    candidates.append(x)
            
            candidates = np.array(candidates)
            
            # Predict at candidates
            mu_cand, var_cand = gp_predict(candidates, X_obs, y_obs, length_scales, sigma_f, sigma_n)
            
            # Compute EI
            y_best = np.min(y_obs)
            ei = expected_improvement(mu_cand, var_cand, y_best, xi=0.01)
            
            # Select best candidate
            best_cand_idx = np.argmax(ei)
            x_next = candidates[best_cand_idx]
            
            # Evaluate
            fx = safe_func(x_next)
            X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
            y_obs = np.append(y_obs, fx)
            
            # Update hyperparameters every 10 iterations
            if (bo_iter + 1) % 10 == 0 and len(X_obs) >= 10:
                n_val = max(2, len(X_obs) // 5)
                X_train = X_obs[:-n_val]
                y_train = y_obs[:-n_val]
                X_val = X_obs[-n_val:]
                best_score = np.inf
                for ls in ls_candidates:
                    ls_vec = np.ones(d) * ls
                    try:
                        mu_val, var_val = gp_predict(X_val, X_train, y_train, ls_vec, sigma_f, sigma_n)
                        var_val = np.maximum(var_val, 1e-12)
                        nll = 0.5 * np.sum(np.log(2 * np.pi * var_val)) + 0.5 * np.sum((y_val - mu_val) ** 2 / var_val)
                        if nll < best_score:
                            best_score = nll
                            length_scales = ls_vec
                    except:
                        continue

        # --- Phase 2: L-BFGS-B polish from multiple starting points ---
        # Collect starting points
        start_points = []
        
        # Best point found
        start_points.append(best_x.copy())
        
        # Top 5 points from observations
        sorted_idx = np.argsort(y_obs)
        for idx in sorted_idx[:5]:
            start_points.append(X_obs[idx].copy())
        
        # A few random points
        for _ in range(3):
            p = rng.uniform(-4, 4, size=d)
            start_points.append(p)
        
        # Remove duplicates
        unique_starts = []
        for p in start_points:
            is_dup = False
            for q in unique_starts:
                if np.allclose(p, q, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(p)
        
        bounds = [(-5, 5)] * d

        # Run L-BFGS-B from each starting point
        for x0 in unique_starts:
            if not time_ok() or calls >= budget - 2:
                break

            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)

            try:
                res = minimize(
                    local_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxls': 50,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'eps': 1e-9
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: Final high-precision polish ---
        if time_ok() and calls < budget - 3:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxls': 100,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'eps': 1e-10
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
