"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Hybrid: (1+1)-ES with 1/5 success rule and restarts, plus DE for exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

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
        # Start at a random point
        x = rng.uniform(-4, 4, size=dim)
        fx = func(x)
        calls_made += 1
        try_update(x, fx)

        # Step size
        sigma = 1.0
        # 1/5 success rule parameters
        success_threshold = 0.2
        step_up = 1.2
        step_down = 1.0 / 1.2

        # Track successes over a window
        window_size = 10
        successes = 0
        steps_in_window = 0

        # Restart logic
        no_improve_count = 0
        max_no_improve = 50
        es_budget = int(budget * 0.5)  # Use 50% of budget for ES

        # Store observations for GP
        obs_x = [x.copy()]
        obs_f = [fx]

        while calls_made < es_budget:
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5, 5)

            candidate_fx = func(candidate)
            calls_made += 1
            obs_x.append(candidate.copy())
            obs_f.append(candidate_fx)

            # Check success
            if candidate_fx <= fx:
                x = candidate
                fx = candidate_fx
                successes += 1
                try_update(x, fx)
                no_improve_count = 0
            else:
                no_improve_count += 1

            steps_in_window += 1

            # Update step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > success_threshold:
                    sigma *= step_up
                elif success_rate < success_threshold:
                    sigma *= step_down
                successes = 0
                steps_in_window = 0

            # Stagnation check: restart if no improvement for max_no_improve steps
            if no_improve_count >= max_no_improve:
                # Try a few random restarts
                for _ in range(3):
                    if calls_made >= es_budget:
                        break
                    x_restart = rng.uniform(-4, 4, size=dim)
                    x_restart_fx = func(x_restart)
                    calls_made += 1
                    obs_x.append(x_restart.copy())
                    obs_f.append(x_restart_fx)
                    if try_update(x_restart, x_restart_fx):
                        x = x_restart
                        fx = x_restart_fx
                        sigma = 1.0
                        no_improve_count = 0
                        break
                # Reset counters
                no_improve_count = 0
                successes = 0
                steps_in_window = 0

        # Phase 2: GP-based Bayesian Optimization for exploration
        bo_budget = budget - calls_made
        if bo_budget > 0 and global_best_x is not None and len(obs_x) >= 2:
            X_obs = np.array(obs_x)
            f_obs = np.array(obs_f)

            # GP hyperparameters
            length_scale = 1.5
            sigma_f = np.std(f_obs) if len(f_obs) > 1 else 1.0
            if sigma_f < 1e-10:
                sigma_f = 1.0
            noise_var = 1e-10

            def gp_predict(X_new, X_train, y_train, l, sigma_y, noise):
                """Predict mean and variance using RBF kernel."""
                X_train = np.atleast_2d(X_train)
                X_new = np.atleast_2d(X_new)
                y_train = np.atleast_1d(y_train)

                # RBF kernel
                def rbf(K1, K2):
                    sq_dists = np.sum(K1 ** 2, axis=1)[:, None] + np.sum(K2 ** 2, axis=1)[None, :] - 2 * K1 @ K2.T
                    return np.exp(-0.5 * sq_dists / (l ** 2))

                K = rbf(X_train, X_train) + noise * np.eye(len(X_train))
                Ks = rbf(X_train, X_new)

                L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                v = np.linalg.solve(L, Ks.T)

                mu = v.T @ alpha
                var = sigma_y ** 2 - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)

                return mu, var

            # Initial: add some random points to help GP exploration
            n_init = min(5, bo_budget // 3)
            for _ in range(n_init):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_new = rng.uniform(-4, 4, size=dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                obs_x.append(x_new.copy())
                obs_f.append(f_new)
                try_update(x_new, f_new)

            # GP-BO loop
            X_train = np.array(obs_x)
            y_train = np.array(obs_f)
            sigma_y = np.std(y_train) if len(y_train) > 1 else 1.0
            if sigma_y < 1e-10:
                sigma_y = 1.0

            for _ in range(bo_budget):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Update GP hyperparameters occasionally
                if _ % 20 == 0 and len(y_train) > 5:
                    X_spread = np.ptp(X_train, axis=0)
                    length_scale = max(0.5, np.mean(X_spread) / 3.0)
                    sigma_y = np.std(y_train) if len(y_train) > 1 else 1.0
                    if sigma_y < 1e-10:
                        sigma_y = 1.0

                # Generate candidate points using EI acquisition
                n_candidates = 50
                candidates = []

                # Random candidates
                for _c in range(n_candidates // 2):
                    candidates.append(rng.uniform(-4, 4, size=dim))

                # Local candidates around current best
                for _c in range(n_candidates // 4):
                    candidates.append(global_best_x + 0.5 * rng.standard_normal(dim))

                # Candidates from interpolation of observed points
                for _c in range(n_candidates // 4):
                    idx1, idx2 = rng.choice(len(X_train), 2, replace=False)
                    t = rng.random()
                    candidates.append((1 - t) * X_train[idx1] + t * X_train[idx2])

                candidates = np.array(candidates)
                candidates = np.clip(candidates, -5, 5)

                # Compute EI for each candidate
                best_f = np.min(y_train)
                mu, var = gp_predict(candidates, X_train, y_train, length_scale, sigma_y, noise_var)
                sigma = np.sqrt(var)

                # Expected Improvement
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (best_f - mu) / sigma
                    z = np.where(sigma > 1e-10, z, 0)
                    EI = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    EI = np.where(sigma > 1e-10, EI, 0)

                # Select best candidate
                best_idx = np.argmax(EI)
                x_new = candidates[best_idx]
                f_new = func(x_new)
                calls_made += 1

                # Update training data
                X_train = np.vstack([X_train, x_new])
                y_train = np.append(y_train, f_new)
                obs_x.append(x_new.copy())
                obs_f.append(f_new)

                if try_update(x_new, f_new):
                    pass

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
                                           'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x
            except Exception:
                pass

            # Additional polish with tighter tolerances if budget remains
            if global_best_x is not None and calls_made < budget and time.time() < deadline:
                remaining2 = budget - calls_made
                if remaining2 > 10:
                    try:
                        result2 = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                           options={'maxiter': remaining2, 'maxfev': remaining2,
                                                    'xatol': 1e-12, 'fatol': 1e-12})
                        if result2.fun < global_best_fx:
                            global_best_fx = result2.fun
                            global_best_x = result2.x
                    except Exception:
                        pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
