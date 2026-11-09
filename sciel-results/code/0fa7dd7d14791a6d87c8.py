"""Bayesian Optimization with GP surrogate and (1+1)-ES local polishing for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(12345)

        # Phase 1: Space-filling initial sampling
        n_init = 25
        X_obs = []
        Y_obs = []
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f_val, x_eval = call_func(x)
            if f_val is not None:
                X_obs.append(x_eval)
                Y_obs.append(f_val)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None, np.inf

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)

        # GP kernel: Matern 5/2
        def matern_kernel(X1, X2, length_scale, variance):
            d = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1))
            sqrt5_d = np.sqrt(5.0) * d / length_scale
            K = variance * (1.0 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        def predict_gp(X_new, X_train, Y_train, length_scale, variance, noise):
            K = matern_kernel(X_train, X_train, length_scale, variance)
            K[np.diag_indices_from(K)] += noise
            K_new = matern_kernel(X_train, X_new, length_scale, variance)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                mean = K_new.T @ alpha
                v = np.linalg.solve(L, K_new)
                var = variance - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                reg = 1e-6
                K_reg = K + reg * np.eye(len(X_train))
                try:
                    alpha = np.linalg.solve(K_reg, Y_train)
                    mean = K_new.T @ alpha
                    var = np.full(len(X_new), variance)
                except np.linalg.LinAlgError:
                    mean = np.full(len(X_new), np.mean(Y_train))
                    var = np.full(len(X_new), np.var(Y_train) + 1e-6)

            return mean, var

        # Phase 2: GP-guided Bayesian Optimization
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6

        Y_mean = np.mean(Y_obs)
        Y_std = np.std(Y_obs) + 1e-10
        Y_norm = (Y_obs - Y_mean) / Y_std

        stagnation_count = 0
        best_f_prev = best_f
        max_gp_points = 80  # Limit GP training set size for speed

        while n_calls < budget and not check_time():
            # Update GP parameters adaptively
            if len(X_obs) >= 10:
                X_range = np.ptp(X_obs, axis=0)
                length_scale = np.max(X_range) / (2.0 * np.sqrt(dim))
                length_scale = np.clip(length_scale, 0.1, 5.0)

                variance = np.var(Y_norm) + 1e-6
                noise = 1e-5 * (np.max(np.abs(Y_norm)) + 1.0)

            # Generate candidate points using multiple strategies
            candidates = []

            # Strategy 1: Sample around best point with decreasing radius
            n_grid = 40
            for _ in range(n_grid):
                radius = 0.3 + 1.5 * np.exp(-0.05 * len(X_obs))
                x_cand = best_x + rng.normal(0, radius, dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)

            # Strategy 2: Pure random for exploration
            for _ in range(15):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)

            # Strategy 3: Latin hypercube-like spread
            for _ in range(15):
                x_cand = rng.uniform(-5, 5, dim)
                x_cand += rng.normal(0, 0.5, dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)

            candidates = np.array(candidates)

            # Compute GP predictions and acquisition function
            if len(X_obs) >= 5:
                # Use subset of observations if too many
                if len(X_obs) > max_gp_points:
                    # Keep best 50% and random 50%
                    n_keep = max_gp_points
                    Y_sorted_idx = np.argsort(Y_obs)
                    n_best_keep = n_keep // 2
                    best_idx = Y_sorted_idx[:n_best_keep]
                    remaining_idx = np.setdiff1d(np.arange(len(X_obs)), best_idx)
                    n_random_keep = n_keep - n_best_keep
                    random_idx = rng.choice(remaining_idx, n_random_keep, replace=False)
                    train_idx = np.concatenate([best_idx, random_idx])
                else:
                    train_idx = np.arange(len(X_obs))

                X_train = X_obs[train_idx]
                Y_train = Y_obs[train_idx]
                Y_train_norm = (Y_train - Y_mean) / Y_std
            else:
                X_train = X_obs
                Y_train = Y_obs
                Y_train_norm = Y_norm

            mean, var = predict_gp(candidates, X_train, Y_train_norm, length_scale, variance, noise)

            # Expected Improvement acquisition
            best_y_norm = np.min(Y_train_norm) if len(Y_train_norm) > 0 else 0.0
            Z = (best_y_norm - mean) / np.sqrt(var)
            EI = (best_y_norm - mean) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
            EI = np.maximum(EI, 0)

            # Mix exploitation and exploration
            n_select = min(4, len(EI))
            top_ei_idx = np.argsort(EI)[-n_select:]

            # Also add some random candidates for diversity
            n_random = max(1, len(EI) // 8)
            random_idx = rng.choice(len(EI), n_random, replace=False)

            selected_idx = np.unique(np.concatenate([top_ei_idx, random_idx]))

            # Evaluate selected candidates
            for idx in selected_idx:
                if check_time() or n_calls >= budget:
                    break
                x_trial = candidates[idx]
                f_val, x_eval = call_func(x_trial)
                if f_val is not None:
                    X_obs = np.vstack([X_obs, x_eval])
                    Y_obs = np.append(Y_obs, f_val)
                    Y_mean = np.mean(Y_obs)
                    Y_std = np.std(Y_obs) + 1e-10
                    Y_norm = (Y_obs - Y_mean) / Y_std

                    if f_val < best_f_prev:
                        best_f_prev = f_val
                        stagnation_count = 0
                    else:
                        stagnation_count += 1

            # Adaptive: if stagnating, increase exploration
            if stagnation_count > 10:
                for _ in range(5):
                    if check_time() or n_calls >= budget:
                        break
                    x_trial = rng.uniform(-5, 5, dim)
                    f_val, x_eval = call_func(x_trial)
                    if f_val is not None:
                        X_obs = np.vstack([X_obs, x_eval])
                        Y_obs = np.append(Y_obs, f_val)
                        Y_mean = np.mean(Y_obs)
                        Y_std = np.std(Y_obs) + 1e-10
                        Y_norm = (Y_obs - Y_mean) / Y_std
                stagnation_count = 0

        # Phase 3: Local refinement with (1+1)-ES around best point
        if n_calls < budget and not check_time():
            sigma = 0.3  # Initial step size
            x_cur = best_x.copy()
            f_cur = best_f

            for _ in range(200):
                if check_time() or n_calls >= budget:
                    break

                # Generate mutant
                x_mut = x_cur + sigma * rng.standard_normal(dim)
                x_mut = np.clip(x_mut, -5, 5)

                f_mut, _ = call_func(x_mut)
                if f_mut is None:
                    break

                if f_mut < f_cur:
                    # Success: accept and adapt sigma
                    x_cur = x_mut
                    f_cur = f_mut
                    sigma *= 1.25  # Increase step size on success
                else:
                    # Failure: decrease step size
                    sigma *= 0.8

                if sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
