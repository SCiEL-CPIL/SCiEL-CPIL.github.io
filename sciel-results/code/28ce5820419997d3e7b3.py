"""Surrogate-assisted Bayesian Optimization for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian optimizer with diverse initialization and adaptive exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        X_obs = []
        y_obs = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_obs.append(x.copy())
            y_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initialization to cover the 101-peak landscape
        n_init = 30
        init_points = []

        # Center point
        init_points.append(np.zeros(dim))

        # Points along each axis
        for d in range(dim):
            for val in [-3.5, -2.0, 0.0, 2.0, 3.5]:
                p = np.zeros(dim)
                p[d] = val
                init_points.append(p)

        # Random points to fill up to n_init
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.5, 4.5, dim))

        init_points = np.array(init_points[:n_init])

        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(init_points[i])

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: GP-based Bayesian Optimization
        # Fit a GP to observed data and use Expected Improvement acquisition
        def fit_gp(X, y):
            """Fit a simple GP with RBF kernel. Returns mean and std functions."""
            n = len(X)
            if n < 2:
                return lambda x: (np.mean(y), 1.0)

            # Standardize inputs for numerical stability
            mu_x = X.mean(axis=0)
            std_x = X.std(axis=0) + 1e-8
            X_std = (X - mu_x) / std_x

            # RBF kernel parameters
            length_scale = 1.5
            variance = 1.0
            noise = 1e-6

            def kernel(X1, X2, ls, var, ns):
                # Squared exponential kernel
                d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :]
                d2 -= 2 * X1 @ X2.T
                d2 = np.maximum(d2, 0)
                return var * np.exp(-d2 / (2 * ls**2)) + ns * np.eye(len(X1))

            K = kernel(X_std, X_std, length_scale, variance, noise)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - np.mean(y)))
            y_mean = np.mean(y)

            def predict(x_new):
                """Predict mean and std at x_new (array of shape (dim,))."""
                x_new_std = (x_new - mu_x) / std_x
                k_star = kernel(x_new_std.reshape(1, -1), X_std, length_scale, variance, 0.0)
                v = np.linalg.solve(L, k_star[0])
                mu = y_mean + v @ alpha
                var_star = variance - np.sum(k_star[0]**2)
                var_star = max(var_star, 1e-10)
                return mu, np.sqrt(var_star)

            return predict

        def expected_improvement(x, best_y, predict_func, xi=0.01):
            """Compute EI acquisition value at x."""
            mu, sigma = predict_func(x)
            if sigma < 1e-8:
                return 0.0
            z = (best_y - mu - xi) / sigma
            ei = (best_y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return max(ei, 0.0)

        # Main BO loop
        bo_calls = 0
        max_bo_calls = budget - calls_made
        stagnation_count = 0
        best_f_bo = best_f

        while bo_calls < max_bo_calls:
            if calls_made >= budget or time.time() > deadline:
                break

            # Fit GP to current observations
            if len(X_obs) < 5:
                # Not enough data for reliable GP, do random exploration
                x_new = rng.uniform(-4.5, 4.5, dim)
            else:
                # Generate candidate points and select best by EI
                n_candidates = 200
                candidates = []

                # Mix of global and local candidates
                n_global = int(n_candidates * 0.4)
                n_local = n_candidates - n_global

                # Global candidates (random)
                for _ in range(n_global):
                    candidates.append(rng.uniform(-4.5, 4.5, dim))

                # Local candidates around best and good points
                if best_x is not None:
                    for _ in range(n_local // 2):
                        candidates.append(best_x + rng.standard_normal(dim) * 0.8)

                # Around other good points
                if len(X_obs) > 1:
                    y_arr = np.array(y_obs)
                    good_indices = np.argsort(y_arr)[:max(3, len(y_arr) // 5)]
                    for idx in good_indices:
                        for _ in range(max(1, n_local // 4)):
                            candidates.append(X_obs[idx] + rng.standard_normal(dim) * 0.5)

                candidates = np.array(candidates)
                candidates = np.clip(candidates, -5.0, 5.0)

                # Fit GP
                predict_func = fit_gp(np.array(X_obs), np.array(y_obs))

                # Compute EI for each candidate
                ei_values = np.array([expected_improvement(c, best_f_bo, predict_func) for c in candidates])

                # Select best candidate
                best_idx = np.argmax(ei_values)
                x_new = candidates[best_idx]

            # Evaluate
            f_new = evaluate(x_new)
            bo_calls += 1

            # Update stagnation
            if f_new < best_f_bo - 1e-8:
                best_f_bo = f_new
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stagnant for too long, inject a random point to escape local optima
            if stagnation_count > 15:
                x_escape = rng.uniform(-4.5, 4.5, dim)
                f_escape = evaluate(x_escape)
                bo_calls += 1
                if f_escape < best_f_bo:
                    best_f_bo = f_escape
                stagnation_count = 0

        # Phase 3: Final polish around the best point
        if calls_made < budget and time.time() < deadline and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.05
            polish_calls = 0
            max_polish = min(20, budget - calls_made)

            while polish_calls < max_polish:
                if calls_made >= budget or time.time() > deadline:
                    break
                y = x_polish + sigma_p * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p = max(0.01, sigma_p * 0.9)
                else:
                    sigma_p = max(0.01, sigma_p * 0.85)
                polish_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
