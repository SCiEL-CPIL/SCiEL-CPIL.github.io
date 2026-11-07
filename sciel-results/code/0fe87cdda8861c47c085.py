"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Hybrid: (1+1)-ES for early exploration, then GP-based Bayesian Optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        # Phase 1: (1+1)-ES with 1/5 success rule
        x = rng.uniform(-4, 4, size=dim)
        fx = func(x)
        calls_made += 1
        try_update(x, fx)

        sigma = 1.0
        success_threshold = 0.2
        step_up = 1.2
        step_down = 1.0 / 1.2

        window_size = 10
        successes = 0
        steps_in_window = 0

        no_improve_count = 0
        max_no_improve = 40
        es_budget = int(budget * 0.3)

        # Collect data points for GP
        data_x = [x.copy()]
        data_f = [fx]

        while calls_made < es_budget:
            if time.time() > deadline:
                break

            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5, 5)

            candidate_fx = func(candidate)
            calls_made += 1
            data_x.append(candidate.copy())
            data_f.append(candidate_fx)

            if candidate_fx <= fx:
                x = candidate
                fx = candidate_fx
                successes += 1
                try_update(x, fx)
                no_improve_count = 0
            else:
                no_improve_count += 1

            steps_in_window += 1

            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > success_threshold:
                    sigma *= step_up
                elif success_rate < success_threshold:
                    sigma *= step_down
                successes = 0
                steps_in_window = 0

            if no_improve_count >= max_no_improve:
                for _ in range(3):
                    if calls_made >= es_budget:
                        break
                    x_restart = rng.uniform(-4, 4, size=dim)
                    x_restart_fx = func(x_restart)
                    calls_made += 1
                    data_x.append(x_restart.copy())
                    data_f.append(x_restart_fx)
                    if try_update(x_restart, x_restart_fx):
                        x = x_restart
                        fx = x_restart_fx
                        sigma = 1.0
                        no_improve_count = 0
                        break
                no_improve_count = 0
                successes = 0
                steps_in_window = 0

        # Phase 2: Bayesian Optimization with Gaussian Process
        # Prepare data
        X_data = np.array(data_x)
        f_data = np.array(data_f)

        # Normalize function values for numerical stability
        f_min = f_data.min()
        f_max = f_data.max()
        f_range = f_max - f_min
        if f_range < 1e-10:
            f_range = 1.0
        f_norm = (f_data - f_min) / f_range

        # GP parameters
        # Matern 5/2 kernel is good for multimodal functions
        def matern52_kernel(X1, X2, length_scale, variance):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            sqrt3_dist = np.sqrt(3) * dist / length_scale
            K = variance * (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
            return K

        def compute_K(X, length_scale, variance, noise):
            K = matern52_kernel(X, X, length_scale, variance)
            K[np.diag_indices_from(K)] += noise
            return K

        def gp_predict(X_new, X_train, f_train, length_scale, variance, noise, alpha):
            """
            Predict mean and variance at X_new given training data.
            alpha = K^{-1} f_train (precomputed)
            """
            K_new = matern52_kernel(X_new, X_train, length_scale, variance)
            mean = K_new @ alpha
            K_new_new = matern52_kernel(X_new, X_new, length_scale, variance)
            var = K_new_new - np.sum(K_new * K_new.T, axis=1)
            var = np.maximum(var, 1e-10)
            return mean, var

        def expected_improvement(mean, var, f_best_norm, xi=0.01):
            """
            Compute Expected Improvement.
            f_best_norm is the best normalized fitness (lowest).
            """
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            z = (f_best_norm - mean - xi) / std
            ei = (f_best_norm - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Initial GP parameters
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6

        # Adaptive parameters for BO
        bo_no_improve = 0
        bo_max_no_improve = 15
        n_candidates = 200
        batch_size = 1

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Recompute alpha (K^{-1} f_train)
            try:
                K = compute_K(X_data, length_scale, variance, noise)
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_norm))
            except np.linalg.LinAlgError:
                # Add more noise if Cholesky fails
                noise *= 10
                continue

            # Generate candidates
            # Mix of random points and points around current best
            candidates = []
            n_random = n_candidates // 2
            n_local = n_candidates - n_random

            # Random candidates
            rand_cands = rng.uniform(-5, 5, size=(n_random, dim))
            candidates.append(rand_cands)

            # Local candidates around best
            local_cands = global_best_x + 0.5 * rng.standard_normal((n_local, dim))
            local_cands = np.clip(local_cands, -5, 5)
            candidates.append(local_cands)

            # Also add some around other good points
            if len(X_data) > 5:
                # Sort by fitness, take top 3
                idx_sorted = np.argsort(f_norm)
                top_idx = idx_sorted[:min(3, len(idx_sorted))]
                for idx in top_idx:
                    extra_cands = X_data[idx] + 0.3 * rng.standard_normal((5, dim))
                    extra_cands = np.clip(extra_cands, -5, 5)
                    candidates.append(extra_cands)

            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, -5, 5)

            # Predict mean and variance
            mean, var = gp_predict(candidates, X_data, f_norm, length_scale, variance, noise, alpha)

            # Compute EI
            f_best_norm = f_norm.min()
            ei = expected_improvement(mean, var, f_best_norm)

            # Select best candidate
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx]

            # Evaluate
            fx_new = func(x_new)
            calls_made += 1

            # Update data
            X_data = np.vstack([X_data, x_new])
            f_data = np.vstack([f_data, fx_new])
            f_norm = np.vstack([f_norm, (fx_new - f_min) / f_range])

            # Update global best
            if try_update(x_new, fx_new):
                bo_no_improve = 0
            else:
                bo_no_improve += 1

            # Adapt GP parameters based on progress
            if bo_no_improve >= bo_max_no_improve:
                # Increase exploration: larger length scale
                length_scale *= 1.5
                length_scale = min(length_scale, 5.0)
                bo_no_improve = 0
                bo_max_no_improve = int(bo_max_no_improve * 1.2)
            elif bo_no_improve == 0:
                # Good progress: slightly reduce length scale for exploitation
                length_scale *= 0.9
                length_scale = max(length_scale, 0.3)

        # Phase 3: Final local refinement using Scipy Nelder-Mead
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = budget - calls_made

            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return 1e10
                calls_made += 1
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining_budget, 'maxfev': remaining_budget,
                                           'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
